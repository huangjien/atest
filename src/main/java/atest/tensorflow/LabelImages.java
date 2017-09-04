package atest.tensorflow;

import java.io.File;
import java.util.List;

public class LabelImages {

    static List<String> labels;


    public static void main(String[] args){

        ModelSetting modelSetting = ModelSetting.getModel_inception_v3();
        //ModelSetting modelSetting = ModelSetting.getModel_inception_graph();
        labels = modelSetting.getLabels();
        for(File file : new File("C:\\temp").listFiles() ){
            if (file.getName().toLowerCase().endsWith("jpg") || file.getName().toLowerCase().endsWith("jpeg")){
                float[] labelProbabilities = modelSetting.executeInceptionGraph(file);
                int bestLabelIdx = maxIndex(labelProbabilities);
                System.out.println(
                        String.format(
                                "File: %s \t\tBEST MATCH: %s (%.2f%% likely)", file.getName(),
                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
            }
        }


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


}
