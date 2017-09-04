package atest.tensorflow;

import org.tensorflow.*;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class ModelSetting {
    String modelFileName;
    String labelsFileName;
    String inputOp;
    String outputOp;
    int h;
    int w;
    float mean = 0f;
    float scale = 224f;
    List<String> list = null;
    byte[] graphDef;
    String modelDir;

    public ModelSetting(String modelDir, String modelFileName, String labelsFileName,
                        String inputOp, String outputOp,
                        int h, int w, float mean, float scale){

        this.modelDir = modelDir;
        this.modelFileName = modelFileName;
        this.labelsFileName = labelsFileName;
        this.inputOp = inputOp;
        this.outputOp = outputOp;
        this.h = h;
        this.w = w;
        this.mean = mean;
        this.scale = scale;
        this.graphDef = readAllBytes(Paths.get(modelDir, modelFileName));
        readLabelsFile();
    }

    public List<String> getLabels(){
        return list;
    }

    public static ModelSetting getModel_inception_v3(){
        return new ModelSetting("./models", "inception_v3_2016_08_28_frozen.pb", "imagenet_slim_labels.txt",
                "input", "InceptionV3/Predictions/Reshape_1", 299, 299, 0f, 255f);
    }

    public static ModelSetting getModel_inception_graph(){
        return new ModelSetting("./models", "tensorflow_inception_graph.pb", "imagenet_comp_graph_label_strings.txt",
                "input", "output", 224, 224, 117f, 255f);
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    public float[] executeInceptionGraph(File file){
        Tensor image = getTensor(file);
        return executeInceptionGraph(image);
    }

    Graph g = null;
    public  float[] executeInceptionGraph(Tensor image) {
        try {
            if(g==null){
                g = new Graph();
                g.importGraphDef(graphDef);
            }

            try (Session s = new Session(g);
                 Tensor result = s.runner().feed(inputOp, image).fetch(outputOp).run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }catch (Exception e){
            System.err.println(e.getMessage());
        }
        return null;
    }

    public Tensor getTensor(File file){
        try (Graph g = new Graph()) {
            GraphBuilder b = new GraphBuilder(g);
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.


            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            final Output input = b.constant("input", readAllBytes(file));
            final Output output =
                    b.div(
                            b.sub(
                                    b.resizeBilinear(
                                            b.expandDims(
                                                    b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                                                    b.constant("make_batch", 0)),
                                            b.constant("size", new int[] {h, w})),
                                    b.constant("mean", mean)),
                            b.constant("scale", scale));
            try (Session s = new Session(g)) {
                return s.runner().fetch(output.op().name()).run().get(0);
            }
        }
    }

    private static byte[] readAllBytes(File path) {
        try {
            return Files.readAllBytes(Paths.get(path.getAbsolutePath()));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            //System.exit(1);
        }
        return null;
    }

    private static byte[] readAllBytes(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            //System.exit(1);
        }
        return null;
    }

    private void readLabelsFile() {
        Path path = Paths.get(modelDir, labelsFileName);
        try {
            list = Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            //System.exit(0);
        }
    }
}
