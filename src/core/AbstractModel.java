package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;
import util.evaluation.Measurement;

/**
 *
 * @author vietan
 */
public abstract class AbstractModel {

    protected String name;
    protected boolean verbose = true;

    public AbstractModel(String name) {
        this.name = name;
    }

    public String getBasename() {
        return this.name;
    }

    public String getName() {
        return this.name;
    }

    public abstract void output(File modelFile);

    public abstract void input(File modelFile);

    public static void log(String msg) {
        System.out.print("[LOG] " + msg);
    }

    public static void logln(String msg) {
        System.out.println("[LOG] " + msg);
    }

    public static void outputPerformances(File perfFile, ArrayList<Measurement> measurements) {
        System.out.println("Outputing performances to " + perfFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(perfFile);
            for (Measurement measurement : measurements) {
                writer.write(measurement.getName() + "\t" + measurement.getValue() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing performances to " + perfFile);
        }
    }

    public static ArrayList<Measurement> inputPerformances(File perfFile) {
        System.out.println("Inputing performances from " + perfFile);
        ArrayList<Measurement> measurements = new ArrayList<>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(perfFile);
            String line;
            String[] sline;
            while ((line = reader.readLine()) != null) {
                sline = line.split("\t");
                measurements.add(new Measurement(sline[0], Double.parseDouble(sline[1])));
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing performances from " + perfFile);
        }
        return measurements;
    }
}
