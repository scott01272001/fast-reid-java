package com.jdai.fastreid;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import com.jdai.fastreid.FastReidEngine.EngineDevice;

@Fork(1)
@BenchmarkMode(Mode.AverageTime)
@Warmup(time = 1, timeUnit = TimeUnit.SECONDS, iterations = 3)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Measurement(time = 10, iterations = 5, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
public class FastReidEngineBenchmark {

    private Mat image;
    private FastReidEngine engine;

    @Param({"test_image.jpg"})
    private String testImage;

    @Param({"duke_sbs_S50_ts.pt"})
    private String modelName;

    @Setup
    public void setup() throws Exception {
        System.setProperty("org.bytedeco.javacpp.pathsfirst", "true");
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        Loader.load(opencv_java.class);

        File tempFile = File.createTempFile("tmp", testImage);
        if (!tempFile.exists()) {
            throw new IOException("Can not create tmp file.");
        }
        Path tempPath = tempFile.toPath();
        InputStream in = this.getClass().getClassLoader().getResourceAsStream(testImage);
        if (in == null) {
            throw new IOException("Can not locate " + testImage);
        }
        Files.copy(in, tempPath, StandardCopyOption.REPLACE_EXISTING);
        String imagePath = tempPath.toFile().getAbsolutePath();

        byte[] bytes = IOUtils.toByteArray(getClass().getClassLoader().getResourceAsStream(modelName));
        File tempModel = File.createTempFile("tmp", ".pt");
        OutputStream outStream = new FileOutputStream(tempModel);
        outStream.write(bytes);
        IOUtils.closeQuietly(outStream);

        image = Imgcodecs.imread(imagePath);

        if (FastReidEngine.checkCuda()) {
            engine = new FastReidEngine(tempModel.getAbsoluteFile().toString(), EngineDevice.GPU, 0);
        } else {
            engine = new FastReidEngine(tempModel.getAbsoluteFile().toString(), EngineDevice.CPU, null);
        }

        if (tempFile.exists()) {
            tempFile.delete();
        }
        if (tempModel.exists()) {
            tempModel.delete();
        }
    }

    @Benchmark
    public float[] detect() throws Exception {
        return engine.extractFeature(image);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder().include(FastReidEngineBenchmark.class.getSimpleName()).build();
        new Runner(opt).run();
    }

}
