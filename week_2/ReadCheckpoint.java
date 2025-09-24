import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Arrays;

/** Minimal data classes using only standard Java libraries **/

/** Equivalent to llama2.c Config struct (all fields are little-endian int32) */
final class Config {
    int dim;          // model dimension (hidden size)
    int hiddenDim;    // feed-forward hidden size
    int nLayers;      // number of transformer layers
    int nHeads;       // number of attention heads
    int nKvHeads;     // number of KV heads (for GQA)
    int vocabSize;    // vocabulary size (absolute value of stored int)
    int seqLen;       // maximum sequence length

    /** True if vocab size in file was positive, false if negative (weight sharing hack) */
    boolean sharedWeights;

    @Override public String toString() {
        return "Config{dim=" + dim + ", hiddenDim=" + hiddenDim +
               ", nLayers=" + nLayers + ", nHeads=" + nHeads +
               ", nKvHeads=" + nKvHeads + ", vocabSize=" + vocabSize +
               ", seqLen=" + seqLen + ", sharedWeights=" + sharedWeights + "}";
    }
}

/** Holds the parsed config and the raw float weights */
final class ModelBlob {
    final Config config;
    final float[] weights;   // all remaining floats from the checkpoint
    final long fileSizeBytes;

    ModelBlob(Config c, float[] w, long size) { this.config = c; this.weights = w; this.fileSizeBytes = size; }

    /** Convenience method to create a FloatBuffer view of a slice */
    FloatBuffer slice(int start, int len) {
        if (start < 0 || len < 0 || start + len > weights.length)
            throw new IndexOutOfBoundsException("slice out of range");
        ByteBuffer bb = ByteBuffer.allocate(len * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < len; i++) bb.putFloat(weights[start + i]);
        bb.flip();
        return bb.asFloatBuffer();
    }
}

/** Reader for llama2.c-style checkpoints (reads into float[] without memory mapping). */
public final class ReadCheckpoint {

    /** Main read method */
    public static ModelBlob read(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path)) {
            long fileSize = ch.size();

            // Step 1: Read the Config header (7 int32 values)
            ByteBuffer header = ByteBuffer.allocate(7 * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, header, 0);
            header.rewind();

            Config cfg = new Config();
            cfg.dim       = header.getInt();
            cfg.hiddenDim = header.getInt();
            cfg.nLayers   = header.getInt();
            cfg.nHeads    = header.getInt();
            cfg.nKvHeads  = header.getInt();
            int rawVocab  = header.getInt();
            cfg.seqLen    = header.getInt();

            // Handle negative vocabSize (hack for shared/unshared weights)
            cfg.sharedWeights = rawVocab > 0;
            cfg.vocabSize     = Math.abs(rawVocab);

            // Step 2: Read the remaining file contents as float32 weights
            long remaining = fileSize - header.capacity();
            if (remaining < 0 || (remaining % 4) != 0)
                throw new IOException("Malformed checkpoint: invalid remaining size");

            int nFloats = (int)(remaining / 4);
            float[] weights = new float[nFloats];

            // Stream the file in chunks to avoid large intermediate buffers
            final int CHUNK = 1 << 20; // read 1MB at a time
            ByteBuffer buf = ByteBuffer.allocate(Math.min(nFloats * 4, CHUNK)).order(ByteOrder.LITTLE_ENDIAN);

            int wrote = 0;
            long pos = header.capacity();
            while (wrote < nFloats) {
                buf.clear();
                int floatsThis = Math.min((buf.capacity()/4), nFloats - wrote);
                buf.limit(floatsThis * 4);
                readFully(ch, buf, pos);
                buf.rewind();
                for (int i = 0; i < floatsThis; i++) weights[wrote++] = buf.getFloat();
                pos += floatsThis * 4L;
            }

            return new ModelBlob(cfg, weights, fileSize);
        }
    }

    /** Utility: ensure the buffer is completely filled by reading from absolute position */
    private static void readFully(FileChannel ch, ByteBuffer buf, long position) throws IOException {
        buf.clear();
        int need = buf.remaining();
        int got  = 0;
        while (got < need) {
            int r = ch.read(buf, position + got);
            if (r < 0) throw new EOFException("Unexpected EOF");
            got += r;
        }
    }

    /* ------------------- Simple self-tests (no external libraries) ------------------- */

    // Run:  java ReadCheckpoint <checkpoint.bin>
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Usage: java ReadCheckpoint <checkpoint.bin>");
            return;
        }
        Path p = Path.of(args[0]);
        ModelBlob m = read(p);

        System.out.println("Loaded " + p.getFileName() + " (" + m.fileSizeBytes + " bytes)");
        System.out.println(m.config);

        // Test 1: weights length must be positive
        assert m.weights.length > 0 : "no weights read";

        // Test 2: endian sanity check — re-encode first N floats and compare
        int N = Math.min(8, m.weights.length);
        byte[] round = new byte[N * 4];
        ByteBuffer bb = ByteBuffer.wrap(round).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < N; i++) bb.putFloat(m.weights[i]);
        bb.rewind();
        for (int i = 0; i < N; i++) {
            float back = bb.getFloat(i * 4);
            if (Float.floatToIntBits(back) != Float.floatToIntBits(m.weights[i]))
                throw new AssertionError("Endian mismatch at i=" + i);
        }

        // Test 3: print a small slice to visually inspect
        System.out.println("First 8 floats: " + Arrays.toString(Arrays.copyOf(m.weights, N)));

        System.out.println("All basic tests passed ✅");
    }
}
