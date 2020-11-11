import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class RawDataToUsableDataConverter {
    public static void main(String[] args){

        // Transform the feature vectors into strictly numerical vectors
        transformFeatureVectorData();

        // Transform the responses into severity indices
        transformResponseDate();
        // Todo: Figure out an equation / way to turn the data into a severity index

        // Match the transformed feature vectors with their corresponding severity index, create 1 csv
        combineFeaturesWithSeverityIndex();

        // Use this file in Matlab to perform Linear Regression.
    }

    /**
     *******************************************************
     * The following methods are used to transform the
     * feature vector data
     *******************************************************
     */

    // Question: What to do about empty data?
    // Question: What other processing / transformation on feature data?
    // - Go through and identify the features you actually care about - DONE SEE FeaturesToKeep.csv
    // NOTE: We will need certain feature information from each county to perform the significance calculation
    public static void transformFeatureVectorData() {
        removeUnwantedFeatureValues();

        /**
         * TODO: Perform any other feature vector transformation here
         */

    }

    public static void removeUnwantedFeatureValues() {
        try {

            BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/StartingFiles/FeaturesToKeep.csv"));
            Set<Integer> wantedFeatures = new HashSet<>();
            String curLine = "";
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(",");
                Integer curFeatureIndex = Integer.parseInt(data[0]);
                wantedFeatures.add(curFeatureIndex);
            }
            csvReader.close();

            csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/StartingFiles/RawCountyDemographics.csv"));
            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/FinalFeatureVectors.csv");
            curLine = "";
            String firstLine = csvReader.readLine();
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(";");
                boolean addVector = true;
                String stringBuilder = data[2];
                // Remove indices 0, 1, 3, 107, 108, 147
                for (int i = 0; i < data.length; i ++) {
                    if (wantedFeatures.contains(i) && i != 2) {
                        // We want to add the remaining fields to
                        stringBuilder += ";" + data[i].trim();
                        if (data[i].trim().equals("")) {
                            addVector = false;
                            // Comment the above line if you wish to allow samples with empty data
                        }
                    } else {
                        // Do nothing, we do not want these fields in the matrix
                    }
                }
                stringBuilder += "\n";
                if (addVector) {
                    csvWriter.append(stringBuilder);
                }
            }
            csvReader.close();
            csvWriter.flush();
            csvWriter.close();

        } catch (IOException d) {
            d.printStackTrace();
        }
    }

    /**
     * End feature vector data transformation methods
     *******************************************************
     */

    /**
     *******************************************************
     * The following methods are used to transform the
     * response data
     *******************************************************
     */

    public static void transformResponseDate() {
        modifyDataFileFormat();
        Set<Integer> countyNumbers = separateDataByCounties();
        calculateSeverityIndices(countyNumbers);
    }

    public static void modifyDataFileFormat() {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/StartingFiles/CovidFileWithDates.csv"));
            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/CovidFileWithModifiedDates.csv");

            SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
            Date day0 = format.parse ( "2020-01-21" );
            long msInDay = 86400000;
            long day0Time = day0.getTime();
            String curLine = "";
            String firstLine = csvReader.readLine();
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(",");

                if (data[0].trim().equals("") ||
                        data[3].trim().equals("") ||
                        data[4].trim().equals("") ||
                        data[5].trim().equals("")) {
                    // System.out.println("Inside empty row");
                    continue;
                }

                Date curDate = format.parse (data[0]);
                long curDateTime = curDate.getTime();

                long timeBetween = curDateTime - day0Time;
                int numberDays = (int)(timeBetween / msInDay);
                String numberDaysString = String.valueOf(numberDays);

                csvWriter.append(numberDaysString);
                csvWriter.append(",");
                csvWriter.append(data[3]);
                csvWriter.append(",");
                csvWriter.append(data[4]);
                csvWriter.append(",");
                csvWriter.append(data[5]);
                csvWriter.append("\n");
            }
            csvReader.close();
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException | ParseException d) {
            d.printStackTrace();
        }
    }

    public static Set<Integer> separateDataByCounties() {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/CovidFileWithModifiedDates.csv"));

            // Q: Which data structure to use as mapping?
            // We are mapping Integer (county#) to a collection of its corresponding data points
            Map<Integer, List<String>> byCountyFiles = new HashMap<>();
            Set<Integer> countiesBeingTracked = new HashSet<>();

            String curLine = "";
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(",");

                Integer countyNumber = Integer.parseInt(data[1]);

                // Using the county number, add the curLine into the ArrayList at that key
                if (byCountyFiles.containsKey(countyNumber)) {
                    byCountyFiles.get(countyNumber).add(curLine);
                } else {
                    List<String> starterList = new ArrayList<>();
                    starterList.add(curLine);
                    countiesBeingTracked.add(countyNumber);
                    byCountyFiles.put(countyNumber, starterList);
                }
            }
            csvReader.close();

            // System.out.println(countiesBeingTracked.size());
            // We now theoretically have a map containing all the individual time series
            // What to do?
            // Convert the cumulative to daily increase
            for (Integer county : countiesBeingTracked) {
                FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/TimeSeries/" + county.toString() + ".csv");
                List<String> curCountyTimeSeries = byCountyFiles.get(county);
                // System.out.println(curCountyTimeSeries.size());
                int prevCases = 0;
                int prevDeaths = 0;
                // #,#,#,#
                for (String curDataLine: curCountyTimeSeries) {
                    String[] data = curDataLine.split(",");
                    // 0 - DayNumber
                    int curDay = Integer.parseInt(data[0]);
                    // 1 - CountyNumber
                    int countyNum = Integer.parseInt(data[1]);
                    // 2 - NumberTotalCases
                    int curCases = Integer.parseInt(data[2]);
                    // 3 - NumberTotalDeaths
                    int curDeaths = Integer.parseInt(data[3]);

                    int deltaCases = curCases - prevCases;
                    int deltaDeaths = curDeaths - prevDeaths;

                    // We now have all the data + the rolling cases

                    csvWriter.append(data[0]);
                    csvWriter.append(",");
                    csvWriter.append(data[1]);
                    csvWriter.append(",");
                    csvWriter.append(String.valueOf(deltaCases));
                    csvWriter.append(",");
                    csvWriter.append(String.valueOf(data[2]));
                    csvWriter.append(",");
                    csvWriter.append(String.valueOf(deltaDeaths));
                    csvWriter.append(",");
                    csvWriter.append(String.valueOf(data[3]));
                    csvWriter.append("\n");

                    prevCases = curCases;
                    prevDeaths = curDeaths;

                }

                csvWriter.flush();
                csvWriter.close();
            }

            return countiesBeingTracked;

        } catch (IOException d) {
            d.printStackTrace();
        }
        return null;
    }

    // Todo: Figure out an equation / way to turn the data into a severity index
    public static void calculateSeverityIndices(Set<Integer> countyNumbers) {
        try {

            // Keep track of the severity indices in a data structure here, going to print after everything done
            List<String> resultList = new ArrayList<>();

            for (Integer curCountyNumber: countyNumbers) {
                BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/TimeSeries/" + curCountyNumber + ".csv"));

                double severityIndex = 0.00;
                String curLine = "";
                while ((curLine = csvReader.readLine()) != null) {
                    String[] data = curLine.split(",");

                    int curDay = Integer.parseInt(data[0]);
                    int countyNum = Integer.parseInt(data[1]);
                    int deltaCases = Integer.parseInt(data[2]);
                    int totalCases = Integer.parseInt(data[3]);
                    int deltaDeaths = Integer.parseInt(data[4]);
                    int totalDeaths = Integer.parseInt(data[5]);

                    /**
                     * TODO: Need to implement a function to calculate severity here
                     */

                    severityIndex = 1;

                }
                csvReader.close();

                String curResponseEntry = curCountyNumber + "," + severityIndex + "\n";
                resultList.add(curResponseEntry);
            }

            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/CovidSeverityIndices.csv");
            for (String response: resultList) {
                csvWriter.append(response);
            }
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException d) {

        }
    }

    /**
     * End response data transformation methods
     *******************************************************
     */

    /**
     *******************************************************
     * The following methods are used to combine the final
     * feature vector set with the appropriate severity
     * level. The resulting file will be usable in further
     * Linear Regression algorithms
     *******************************************************
     */

    public static void combineFeaturesWithSeverityIndex() {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/FinalFeatureVectors.csv"));
            Map<Integer, String> countyDemographicsMap = new HashMap<>();
            String curLine = "";
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(";");
                int countyNumber = Integer.parseInt(data[0]);
                countyDemographicsMap.put(countyNumber, curLine);
            }
            csvReader.close();

            csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/CovidSeverityIndices.csv"));
            Map<Integer, Double> countyCovidSeverityMap = new HashMap<>();
            List<Integer> countyNumberList = new ArrayList<>();
            curLine = "";
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(",");
                int countyNumber = Integer.parseInt(data[0]);
                countyNumberList.add(countyNumber);
                countyCovidSeverityMap.put(countyNumber, Double.parseDouble(data[1]));
            }
            csvReader.close();

            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/FinalCombinedFeaturesWithSeverity.csv");

            for (Integer countyNumber: countyNumberList) {
                double curCountySeverityIndex = countyCovidSeverityMap.get(countyNumber);
                String curCountyFeatureVector = countyDemographicsMap.get(countyNumber);

                if (curCountyFeatureVector != null) {
                    csvWriter.append(String.valueOf(curCountySeverityIndex));
                    csvWriter.append(";");
                    csvWriter.append(curCountyFeatureVector);
                    csvWriter.append("\n");
                }

            }
            csvWriter.flush();
            csvWriter.close();

        } catch (IOException d) {
            d.printStackTrace();
        }
    }

    /**
     * End feature / response combination methods
     *******************************************************
     */

}
