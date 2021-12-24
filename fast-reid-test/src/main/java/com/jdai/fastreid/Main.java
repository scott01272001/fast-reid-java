package com.jdai.fastreid;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import com.jdai.fastreid.FastReidEngine.EngineDevice;

public class Main {

    public static void main(String[] args) throws Exception {
        String modelPath = null;
        String imagePath = null;
        String device = "cpu";
        Integer index = 0;

        Options options = new Options();
        options.addOption(new Option("h", "help", false, "how to use."));
        options.addOption(Option.builder("m").longOpt("model").argName("model").hasArg().required(true).desc("model path").build());
        options.addOption(Option.builder("i").longOpt("input").argName("input").hasArg().required(true).desc("input image").build());
        options.addOption(Option.builder("d").longOpt("device").argName("device").hasArg().required(false).desc("device").build());
        options.addOption(Option.builder("index").longOpt("gpu_index").argName("index").hasArg().type(Integer.class).required(false).desc("gpu index").build());

        CommandLine cmd;
        CommandLineParser parser = DefaultParser.builder().build();
        HelpFormatter helper = new HelpFormatter();

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("m")) {
                modelPath = cmd.getOptionValue("model");
            }
            if (cmd.hasOption("i")) {
                imagePath = cmd.getOptionValue("input");
            }
            if (cmd.hasOption("d")) {
                device = cmd.getOptionValue("device");
            }
            if (cmd.hasOption("index")) {
                index = (Integer) cmd.getParsedOptionValue("index");
            }
        } catch (Exception e) {
            helper.printHelp("Usage:", options);
            System.exit(0);
        }

        FastReidEngine engine = null;
        if (device.equals("cpu")) {
            engine = new FastReidEngine(modelPath, EngineDevice.CPU, null);
        }
        if (device.equals("gpu")) {
            engine = new FastReidEngine(modelPath, EngineDevice.GPU, index);
        }

        Mat image = Imgcodecs.imread(imagePath);

        float[] features = engine.extractFeature(image);

        for (int i = 0; i < features.length; i++) {
            System.out.print(features[i] + ", ");
        }
    }

}
