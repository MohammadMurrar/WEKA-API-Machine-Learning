import weka.core.*;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.evaluation.Evaluation;
import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {

        DataSource source0 = new DataSource("G:\\My Drive\\UNIVERSITY\\4th year\\1st Semester\\COMP338\\P3\\DataSet\\Height_Weight.arff");
        Instances dataset = source0.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        System.out.println();
        System.out.println("####### Statistics of Height and Weight Before Convert Units #######");
        printStatistics(dataset);
        System.out.println("####################################################################");
        System.out.println();
        System.out.println("####### Statistics of Height and Weight After Convert Units ########");
        convertUnits(dataset);
        printStatistics(dataset);
        System.out.println("####################################################################");
        System.out.println();
        System.out.println();

        /* Saving the Converted Data in new file */
//		Instances convertedDataSet = datasetNotConverted;
//		CSVSaver csvSaver = new CSVSaver();
//        csvSaver.setInstances(convertedDataSet);
//		try {
//            csvSaver.setFile(new File("C:\\Users\\momur\\Downloads\\Converted_Height_Weight.csv"));
//            csvSaver.writeBatch();
//		    System.out.println("CSV with Convert units data file saved successfully.");
//		} catch (Exception e) {
//		    e.printStackTrace();
//		}


        DataSource source = new DataSource("G:\\My Drive\\UNIVERSITY\\4th year\\1st Semester\\COMP338\\P3\\DataSet\\Converted_Height_Weight.arff");
        Instances convertedDataset = source0.getDataSet();

        System.out.println("######## Model One Performance Statistics  ########");
        Instances m1Dataset = selectRandomSubset(convertedDataset, 100);
        evaluateModel(m1Dataset, "Model M1");
        System.out.println("###################################################");

        System.out.println();

        System.out.println("######## Model Two Performance Statistics  ########");
        Instances m2Dataset = selectRandomSubset(convertedDataset, 1000);
        evaluateModel(m2Dataset, "Model M2");
        System.out.println("###################################################");

        System.out.println();

        System.out.println("####### Model Three Performance Statistics  ########");
        Instances m3Dataset = selectRandomSubset(convertedDataset, 5000);
        evaluateModel(m3Dataset, "Model M3");
        System.out.println("####################################################");

        System.out.println();

        System.out.println("######## Model Four Performance Statistics  ########");
        Instances m4Dataset = convertedDataset;
        evaluateModel(m4Dataset, "Model M4");
        System.out.println("####################################################");

    }

    private static void convertUnits(Instances data) {
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);

            double h = instance.value(1) * 2.54;
            instance.setValue(1, h);

            double w = instance.value(2) * 0.453592;
            instance.setValue(2, w);
        }
    }

    private static void printStatistics(Instances data) {
        for (int i = 1; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            System.out.println("Statistics of " + attribute.name() + ":");
            System.out.println("Mean: " + data.meanOrMode(attribute));

            double[] values = data.attributeToDoubleArray(i);
            Arrays.sort(values);

            double median;
            int middle = values.length / 2;
            if (values.length % 2 == 0) {
                median = (values[middle - 1] + values[middle]) / 2;
            } else {
                median = values[middle];
            }
            System.out.println("Median: " + median);

            System.out.println("Standard Deviation: " + Math.sqrt(data.variance(attribute)));


            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < data.numInstances(); j++) {
                double val = data.instance(j).value(attribute);
                if (val < min)
                    min = val;
                if (val > max)
                    max = val;
            }
            System.out.println("Minimun Value: " + min);
            System.out.println("Maximum Value: " + max);
            System.out.println();
        }
    }

    private static Instances selectRandomSubset(Instances dataset, int size) throws Exception {
        dataset.randomize(new Random());
        return new Instances(dataset, 0, size);
    }

    private static void evaluateModel(Instances subset, String modelName) throws Exception {
        // Set the class index to the last attribute

        subset.setClassIndex(subset.numAttributes() - 1);

        // Splitting the data
        int trainSize = (int) Math.round(subset.numInstances() * 0.7);
        int testSize = subset.numInstances() - trainSize;

        Instances train = new Instances(subset, 0, trainSize);
        Instances test = new Instances(subset, trainSize, testSize);

        // Building the model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(train);

        // Evaluating the model
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(model, test);

        System.out.println(modelName + " Evaluation:");
        System.out.println("Correlation coefficient: " + evaluation.correlationCoefficient());
        System.out.println("Mean absolute error: " + evaluation.meanAbsoluteError());
        System.out.println("Root mean squared error: " + evaluation.rootMeanSquaredError());
        System.out.println("Relative absolute error: " + evaluation.relativeAbsoluteError() + "%");
        System.out.println("Root relative squared error: " + evaluation.rootRelativeSquaredError() + "%");
        System.out.println("Number Of Instances: " + test.size());
        System.out.println();
    }
}
