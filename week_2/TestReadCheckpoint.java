import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;

/**
 * 测试套件用于测试ReadCheckpoint类
 * 包含单元测试、集成测试和边界条件测试
 */
public class TestReadCheckpoint {
    
    private static int testsPassed = 0;
    private static int testsFailed = 0;
    
    public static void main(String[] args) {
        System.out.println("=== ReadCheckpoint 测试套件 ===\n");
        
        // 运行所有测试
        testConfigClass();
        testModelBlobClass();
        testSliceMethod();
        testReadFullyMethod();
        testErrorHandling();
        testWithMockData();
        
        // 如果有真实检查点文件，测试它
        if (args.length > 0) {
            testWithRealCheckpoint(args[0]);
        } else {
            System.out.println("提示: 提供检查点文件路径作为参数来测试真实文件");
            System.out.println("用法: java TestReadCheckpoint <checkpoint.bin>");
        }
        
        // 输出测试结果
        System.out.println("\n=== 测试结果 ===");
        System.out.println("通过: " + testsPassed);
        System.out.println("失败: " + testsFailed);
        System.out.println("总计: " + (testsPassed + testsFailed));
        
        if (testsFailed == 0) {
            System.out.println("✅ 所有测试通过！");
        } else {
            System.out.println("❌ 有测试失败，请检查上述错误信息");
        }
    }
    
    /**
     * 测试Config类的功能
     */
    private static void testConfigClass() {
        System.out.println("测试Config类...");
        
        try {
            Config config = new Config();
            config.dim = 512;
            config.hiddenDim = 2048;
            config.nLayers = 8;
            config.nHeads = 8;
            config.nKvHeads = 8;
            config.vocabSize = 32000;
            config.seqLen = 1024;
            config.sharedWeights = true;
            
            String toString = config.toString();
            assert toString.contains("dim=512") : "toString方法应该包含dim信息";
            assert toString.contains("hiddenDim=2048") : "toString方法应该包含hiddenDim信息";
            
            System.out.println("  ✅ Config类测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ Config类测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 测试ModelBlob类的功能
     */
    private static void testModelBlobClass() {
        System.out.println("测试ModelBlob类...");
        
        try {
            Config config = new Config();
            config.dim = 256;
            
            float[] weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            ModelBlob blob = new ModelBlob(config, weights, 1000L);
            
            assert blob.config == config : "config应该正确设置";
            assert blob.weights == weights : "weights应该正确设置";
            assert blob.fileSizeBytes == 1000L : "fileSizeBytes应该正确设置";
            
            System.out.println("  ✅ ModelBlob类测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ ModelBlob类测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 测试slice方法
     */
    private static void testSliceMethod() {
        System.out.println("测试slice方法...");
        
        try {
            Config config = new Config();
            float[] weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            ModelBlob blob = new ModelBlob(config, weights, 1000L);
            
            // 测试正常切片
            FloatBuffer slice = blob.slice(2, 3);
            assert slice.capacity() == 3 : "slice容量应该正确";
            assert slice.get(0) == 3.0f : "第一个元素应该正确";
            assert slice.get(1) == 4.0f : "第二个元素应该正确";
            assert slice.get(2) == 5.0f : "第三个元素应该正确";
            
            // 测试边界条件
            try {
                blob.slice(-1, 2);
                throw new AssertionError("应该抛出IndexOutOfBoundsException");
            } catch (IndexOutOfBoundsException e) {
                // 预期的异常
            }
            
            try {
                blob.slice(0, 10);
                throw new AssertionError("应该抛出IndexOutOfBoundsException");
            } catch (IndexOutOfBoundsException e) {
                // 预期的异常
            }
            
            System.out.println("  ✅ slice方法测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ slice方法测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 测试readFully方法（通过反射）
     */
    private static void testReadFullyMethod() {
        System.out.println("测试readFully方法...");
        
        try {
            // 创建临时文件
            Path tempFile = Files.createTempFile("test", ".bin");
            byte[] testData = {1, 2, 3, 4, 5, 6, 7, 8};
            Files.write(tempFile, testData);
            
            // 使用反射访问私有方法
            java.lang.reflect.Method readFullyMethod = ReadCheckpoint.class.getDeclaredMethod(
                "readFully", FileChannel.class, ByteBuffer.class, long.class);
            readFullyMethod.setAccessible(true);
            
            try (FileChannel channel = FileChannel.open(tempFile)) {
                ByteBuffer buffer = ByteBuffer.allocate(4);
                readFullyMethod.invoke(null, channel, buffer, 0);
                
                buffer.rewind();
                assert buffer.get() == 1 : "第一个字节应该正确";
                assert buffer.get() == 2 : "第二个字节应该正确";
                assert buffer.get() == 3 : "第三个字节应该正确";
                assert buffer.get() == 4 : "第四个字节应该正确";
            }
            
            Files.deleteIfExists(tempFile);
            System.out.println("  ✅ readFully方法测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ readFully方法测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 测试错误处理
     */
    private static void testErrorHandling() {
        System.out.println("测试错误处理...");
        
        try {
            // 测试不存在的文件
            try {
                ReadCheckpoint.read(Path.of("nonexistent.bin"));
                throw new AssertionError("应该抛出IOException");
            } catch (IOException e) {
                // 预期的异常
            }
            
            // 测试空文件
            Path emptyFile = Files.createTempFile("empty", ".bin");
            try {
                ReadCheckpoint.read(emptyFile);
                throw new AssertionError("应该抛出IOException");
            } catch (IOException e) {
                // 预期的异常
            }
            Files.deleteIfExists(emptyFile);
            
            System.out.println("  ✅ 错误处理测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ 错误处理测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 使用模拟数据测试
     */
    private static void testWithMockData() {
        System.out.println("测试模拟数据...");
        
        try {
            // 创建模拟检查点文件
            Path mockFile = createMockCheckpoint();
            
            ModelBlob result = ReadCheckpoint.read(mockFile);
            
            // 验证配置
            assert result.config.dim == 256 : "dim应该正确";
            assert result.config.hiddenDim == 512 : "hiddenDim应该正确";
            assert result.config.nLayers == 4 : "nLayers应该正确";
            assert result.config.nHeads == 8 : "nHeads应该正确";
            assert result.config.nKvHeads == 8 : "nKvHeads应该正确";
            assert result.config.vocabSize == 1000 : "vocabSize应该正确";
            assert result.config.seqLen == 128 : "seqLen应该正确";
            assert result.config.sharedWeights : "sharedWeights应该为true";
            
            // 验证权重
            assert result.weights.length > 0 : "应该有权重数据";
            assert result.fileSizeBytes > 0 : "文件大小应该大于0";
            
            Files.deleteIfExists(mockFile);
            System.out.println("  ✅ 模拟数据测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ 模拟数据测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 使用真实检查点文件测试
     */
    private static void testWithRealCheckpoint(String checkpointPath) {
        System.out.println("测试真实检查点文件: " + checkpointPath);
        
        try {
            Path path = Path.of(checkpointPath);
            if (!Files.exists(path)) {
                System.out.println("  ❌ 文件不存在: " + checkpointPath);
                testsFailed++;
                return;
            }
            
            ModelBlob result = ReadCheckpoint.read(path);
            
            System.out.println("  📊 配置信息:");
            System.out.println("    - 模型维度: " + result.config.dim);
            System.out.println("    - 隐藏维度: " + result.config.hiddenDim);
            System.out.println("    - 层数: " + result.config.nLayers);
            System.out.println("    - 注意力头数: " + result.config.nHeads);
            System.out.println("    - KV头数: " + result.config.nKvHeads);
            System.out.println("    - 词汇表大小: " + result.config.vocabSize);
            System.out.println("    - 序列长度: " + result.config.seqLen);
            System.out.println("    - 共享权重: " + result.config.sharedWeights);
            System.out.println("    - 权重数量: " + result.weights.length);
            System.out.println("    - 文件大小: " + result.fileSizeBytes + " 字节");
            
            System.out.println("  ✅ 真实检查点文件测试通过");
            testsPassed++;
        } catch (Exception e) {
            System.out.println("  ❌ 真实检查点文件测试失败: " + e.getMessage());
            testsFailed++;
        }
    }
    
    /**
     * 创建模拟检查点文件
     */
    private static Path createMockCheckpoint() throws IOException {
        Path tempFile = Files.createTempFile("mock_checkpoint", ".bin");
        
        try (FileOutputStream fos = new FileOutputStream(tempFile.toFile());
             DataOutputStream dos = new DataOutputStream(fos)) {
            
            // 写入配置头（7个int32，小端序）
            dos.writeInt(Integer.reverseBytes(256));   // dim
            dos.writeInt(Integer.reverseBytes(512));   // hiddenDim
            dos.writeInt(Integer.reverseBytes(4));     // nLayers
            dos.writeInt(Integer.reverseBytes(8));     // nHeads
            dos.writeInt(Integer.reverseBytes(8));     // nKvHeads
            dos.writeInt(Integer.reverseBytes(1000));  // vocabSize (正数，表示不共享权重)
            dos.writeInt(Integer.reverseBytes(128));   // seqLen
            
            // 写入一些模拟权重数据
            for (int i = 0; i < 100; i++) {
                dos.writeFloat(Float.intBitsToFloat(Integer.reverseBytes(Float.floatToIntBits(i * 0.1f))));
            }
        }
        
        return tempFile;
    }
}
