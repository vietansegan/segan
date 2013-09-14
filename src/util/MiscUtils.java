/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

/**
 *
 * @author vietan
 */
public class MiscUtils {

    protected static final NumberFormat formatter = new DecimalFormat("###.###");

    public static double[] flatten2DArray(double[][] array) {
        int length = 0;
        for (int i = 0; i < array.length; i++) {
            length += array[i].length;
        }
        double[] flattenArray = new double[length];
        int count = 0;
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                flattenArray[count++] = array[i][j];
            }
        }
        return flattenArray;
    }

    public static String listToString(List<Double> list) {
        if (list.isEmpty()) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(list.get(0)));
        for (int i = 1; i < list.size(); i++) {
            str.append(", ").append(formatDouble(list.get(i)));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(double[] array) {
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(float[] array) {
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(int[] array) {
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToSVMLightString(int[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String arrayToSVMLightString(float[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String arrayToSVMLightString(double[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String formatDouble(double value) {
        return formatter.format(value);
    }
}
