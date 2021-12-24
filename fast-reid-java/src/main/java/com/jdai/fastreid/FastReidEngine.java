package com.jdai.fastreid;

import static org.bytedeco.pytorch.global.torch.*;
import java.io.File;
import java.io.FileNotFoundException;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_java;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.DeviceType;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class FastReidEngine {
    private static final Size SIZE = new Size(128, 256);
    private JitModule model;
    private Device device;

    public static enum EngineDevice {
        CPU, GPU
    }

    static {
        System.setProperty("org.bytedeco.javacpp.pathsfirst", "true");
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        Loader.load(opencv_java.class);
    }

    public static boolean checkCuda() {
        return cuda_is_available();
    }

    public FastReidEngine(String modelPath, EngineDevice engineDevice, Integer gpuId/* =-1 */) throws Exception {
        try {
            File f = new File(modelPath);
            if (!f.exists()) {
                throw new FileNotFoundException("Model not found.");
            }

            if (engineDevice.equals(EngineDevice.CPU)) {
                device = new Device(DeviceType.CPU);
            } else {
                if (cuda_is_available()) {
                    if (gpuId == null) {
                        gpuId = 0;
                    }
                    device = new Device(DeviceType.CUDA, (byte) (int) gpuId);
                } else {
                    throw new Exception("Cuda is not available.");
                }
            }

            model = load(modelPath, new DeviceOptional(device));
            model.eval();

            if (model.modules().size() == 0) {
                throw new Exception("Faile to create model instance.");
            }
        } catch (Exception e) {
            throw e;
        }
    }

    public float[] extractFeature(Mat image) {
        if (model == null) {
            return null;
        }

        float[] features;

        Tensor input = preprocess(image);

        try (PointerScope scope = new PointerScope()) {
            input = input.to(device, ScalarType.Float);
            IValueVector inputs = new IValueVector();
            inputs.push_back(new IValue(input));

            Tensor output = this.model.forward(inputs).toTensor();
            Tensor normTensor = normalize(output);
            normTensor = normTensor.to(new Device(DeviceType.CPU), ScalarType.Float);

            features = new float[(int) normTensor.size(1)];
            FloatPointer fp = new FloatPointer(normTensor.data_ptr());
            fp.get(features);
        }

        return features;
    }

    private Tensor preprocess(Mat image) {
        Mat tmpImage = new Mat();
        Imgproc.resize(image, tmpImage, SIZE, 0, 0, Imgproc.INTER_CUBIC);
        image = tmpImage.clone();

        Imgproc.cvtColor(image, tmpImage, Imgproc.COLOR_RGB2BGR);
        image = tmpImage.clone();

        image.convertTo(tmpImage, CvType.CV_32FC3);
        image = tmpImage.clone();

        float[] buffer = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, buffer);
        FloatPointer pointer = new FloatPointer(buffer);

        long[] sizes = new long[] {(long) image.size().height, (long) image.size().width, (long) image.channels()};
        Tensor tensor = from_blob(pointer, sizes).permute(2, 0, 1).unsqueeze(0);

        return tensor;
    }

}
