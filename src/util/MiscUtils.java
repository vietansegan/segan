package util;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author vietan
 */
public class MiscUtils {

    protected static final NumberFormat formatter = new DecimalFormat("###.###");

    public static ArrayList<RankingItem<Integer>> getRankingList(double[] scores) {
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < scores.length; ii++) {
            rankItems.add(new RankingItem<Integer>(ii, scores[ii]));
        }
        Collections.sort(rankItems);
        return rankItems;
    }

    public static void incrementMap(HashMap<Integer, Integer> map, Integer key) {
        Integer count = map.get(key);
        if (count == null) {
            map.put(key, 1);
        } else {
            map.put(key, count + 1);
        }
    }

    public static void incrementMap(HashMap<String, Integer> map, String key) {
        Integer count = map.get(key);
        if (count == null) {
            map.put(key, 1);
        } else {
            map.put(key, count + 1);
        }
    }

    public static int getRoundStepSize(int total, int numSteps) {
        int stepSize = (int) Math.pow(10, (int) Math.log10(total / numSteps));
        return stepSize;
    }

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
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(float[] array) {
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(int[] array) {
        if (array.length == 0) {
            return "[]";
        }
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
