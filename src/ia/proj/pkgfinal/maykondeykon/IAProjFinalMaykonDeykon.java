package ia.proj.pkgfinal.maykondeykon;

import com.opencsv.CSVWriter;
import java.io.FileWriter;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author maykon
 */
public class IAProjFinalMaykonDeykon {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("./dados/heart-statlog.arff");

        for (int i = 0; i < 10; i++) {
            Classifier algoritmo = new NaiveBayes();
            executaCrossValidate(source, algoritmo, 10);
        }
        for (int i = 0; i < 10; i++) {
            Classifier algoritmo = new J48();
            executaCrossValidate(source, algoritmo, 10);
        }
        for (int i = 0; i < 10; i++) {
            Classifier algoritmo = new BIFReader();
            executaCrossValidate(source, algoritmo, 10);
        }

    }

    static void executaCrossValidate(DataSource source, Classifier algoritmo, int numPartes) throws Exception {
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);
        Evaluation eval = new Evaluation(instances);

        String nomeDataSet = source.getDataSet().relationName();
        String nomeAlgoritmo = algoritmo.getClass().getSimpleName();

        long start = System.currentTimeMillis();
        eval.crossValidateModel(algoritmo, instances, numPartes, new Random(1));
        long elapsed = System.currentTimeMillis() - start;

        System.out.println("Dataset: " + nomeDataSet);
        System.out.println("Algoritmo: " + nomeAlgoritmo);
        System.out.println(eval.toSummaryString("\nResults\n\n", false));
        System.out.println("Tempo de execução: " + elapsed + " milisegundos");
        System.out.println("\n--------------------------------------------------------------------- \n");

        exportToCSV(nomeDataSet, nomeAlgoritmo, eval, elapsed);

    }

    static void exportToCSV(String dataset, String algoritmo, Evaluation eval, long tempo) throws IOException {
        String arquivo = "src/dados/resultados.csv";
        try (CSVWriter writer = new CSVWriter(new FileWriter(arquivo, true), '\t')) {
            String[] entries = {dataset, algoritmo, eval.pctCorrect() + "", eval.pctIncorrect() + "", tempo + ""};
            writer.writeNext(entries);

        }
    }
}
