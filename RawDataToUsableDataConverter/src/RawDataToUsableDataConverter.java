import java.io.*;
import java.lang.reflect.Array;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class RawDataToUsableDataConverter {
    public static void main(String[] args){

        // Transform the feature vectors into strictly numerical vectors
        transformFeatureVectorData();

        // Transform the responses into severity indices
        transformResponseDate();

        // Match the transformed feature vectors with their corresponding severity index, create 1 csv
        combineFeatureVectorAndResponseData();

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

    private static Map<Integer, Integer> countyPopulations;

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

            // Load in all the populations based on census files
            Map<Integer, Integer> rawCountyPopulations = new HashMap<>();
            csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/StartingFiles/CensusCountyPopulations.csv"));
            String curCountyPopulationLine = "";
            String firstCountyPopulationLine = csvReader.readLine();
            while ((curCountyPopulationLine = csvReader.readLine()) != null) {
                String[] data = curCountyPopulationLine.split(",");
                String stateCode = data[3];
                String countyCode = data[4];
                int fips = Integer.parseInt(stateCode + countyCode);
                int countyPopulation = Integer.parseInt(data[18]);
                rawCountyPopulations.put(fips, countyPopulation);
            }
            csvReader.close();

            // Read in the demographics for each of the counties in the demographics file
            csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/StartingFiles/RawCountyDemographics.csv"));

            File fileName = new File("RawDataToUsableDataConverter/src/FinalFeatureVectors.csv");
            if (fileName.exists()) {
                fileName.delete();
            }

            countyPopulations = new HashMap<>();

            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/FinalFeatureVectors.csv");
            curLine = "";
            String firstLine = csvReader.readLine();
            //Set<Integer> missingSet = new HashSet<>();
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(";");

                boolean addVector = true;
                String stringBuilder = data[2];
                int population = Integer.parseInt(data[32]);
                int countyNumber = Integer.parseInt(data[2]);
                // Remove indices 0, 1, 3, 107, 108, 147
                for (int i = 0; i < data.length; i ++) {
                    if (wantedFeatures.contains(i) && i != 2) {
                        if (i == 32) {
                            if (rawCountyPopulations.get(countyNumber) == null) {
                                addVector = false;
                            } else {
                                stringBuilder += ";" + rawCountyPopulations.get(countyNumber);
                            }
                        } else {
                            // We want to add the remaining fields to
                            stringBuilder += ";" + data[i].trim();
                            if (data[i].trim().equals("")) {
                                //missingSet.add(i);
                                //System.out.println(i);
                                addVector = false;
                                // Comment the above line if you wish to allow samples with empty data
                            }
                        }
                    } else {
                        // Do nothing, we do not want these fields in the matrix
                    }
                }
                stringBuilder += "\n";
                if (addVector) {
                    csvWriter.append(stringBuilder);
//                    countyPopulations.put(countyNumber, population);
                    countyPopulations.put(countyNumber, rawCountyPopulations.get(countyNumber));
                }
            }
            csvReader.close();
            csvWriter.flush();
            csvWriter.close();

            //System.out.println("Missing Features:");
            //System.out.println(missingSet.toString());

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
        setupTempDirectoryStructure();
        modifyDataFileFormat();
        Set<Integer> countyNumbers = separateDataByCounties();
        calculateSeverityIndices(countyNumbers);
        tearDownTempDirectoryStructure(countyNumbers);
    }

    private static void setupTempDirectoryStructure() {

        String directoryName = "RawDataToUsableDataConverter/src/TimeSeries";
        File directory = new File(directoryName);
        if (! directory.exists()){
            directory.mkdir();
            // If you require it to make the entire directory path including parents,
            // use directory.mkdirs(); here instead.
        }
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
                System.out.println(curLine);
                if (data.length < 6) {
                    continue;
                } else if (data[0].trim().equals("") ||
                        data[3].trim().equals("") ||
                        data[4].trim().equals("") ||
                        data[5].trim().equals("")) {
                    // System.out.println("Inside empty row");
                    continue;
                }

                Date curDate = format.parse(data[0]);
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
                Integer countyPopulation = 0;

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

    public static void calculateSeverityIndices(Set<Integer> countyNumbers) {
        try {
            // Keep track of the severity indices in a data structure here, going to print after everything done
            List<String> resultList = new ArrayList<>();
            Map<Integer, ArrayList<Double>> totalCasesPerCapita = new HashMap<>();
            Set<Integer> weekNumberSet = new HashSet<>();

            for (Integer curCountyNumber: countyNumbers) {
                List<WeeklyDataPoint> weeklyCasesPerCapita = getWeeklyCasesPerCapitaList(curCountyNumber);
                // Go through all counties and add their cases/capita value to the list indexed by their week number
                if (weeklyCasesPerCapita != null) {
                    for (WeeklyDataPoint curWeekData : weeklyCasesPerCapita) {
                        int curWeekNumber = curWeekData.getWeekNumber();
                        double curCasesPerCapita = curWeekData.getCasesPerCapita();

                        if (!totalCasesPerCapita.containsKey(curWeekNumber)) {
                            ArrayList<Double> newWeekList = new ArrayList<>();
                            newWeekList.add(curCasesPerCapita);
                            totalCasesPerCapita.put(curWeekNumber, newWeekList);
                            weekNumberSet.add(curWeekNumber);
                        } else {
                            ArrayList<Double> existingWeekList = totalCasesPerCapita.get((curWeekNumber));
                            existingWeekList.add(curCasesPerCapita);
                            totalCasesPerCapita.replace(curWeekNumber, existingWeekList);
                        }
                    }
                }
            }

            Map<Integer, Double> weeklyMeans = new HashMap<>();
            Map<Integer, Double> weeklyVariances = new HashMap<>();

            // 1. Sum all the values for a particular week
            // 2. Determine the Mean
            // 3. Determine the Variance

            // For Each Week
            for (Integer week : weekNumberSet) {
                ArrayList<Double> curWeekDataPoints = totalCasesPerCapita.get(week);
                if (curWeekDataPoints != null) { // defense
                    int numCounties = curWeekDataPoints.size();

                    // 1. Sum all the values for a particular week
                    // 2. Determine the Mean
                    double meanSum = 0;
                    for (Double curData : curWeekDataPoints) {
                        meanSum += curData;
                    }
                    double mean = meanSum / numCounties;
                    if (!weeklyMeans.containsKey(week)) {
                        weeklyMeans.put(week, mean);
                    } else {

                    }

                    // 3. Determine the Variance
                    double varianceSum = 0;
                    for (Double curDate : curWeekDataPoints) {
                        double curVariance = Math.pow((curDate - mean), 2.00);
                        varianceSum += curVariance;
                    }
                    double variance = varianceSum / numCounties;
                    if (!weeklyVariances.containsKey(week)) {
                        weeklyVariances.put(week, variance);
                    } else {

                    }
                }
            }

            // 4. Go back through each county and determine the number of standard deviations from mean
            for (Integer curCountyNumber: countyNumbers) {
                List<WeeklyDataPoint> weeklyCasesPerCapita = getWeeklyCasesPerCapitaList(curCountyNumber);
                if (weeklyCasesPerCapita != null) {
                    int totalNumWeeks = weeklyCasesPerCapita.size();
                    double zScoreSum = 0.00;

                    for (WeeklyDataPoint week : weeklyCasesPerCapita) {
                        int weekNumber = week.getWeekNumber();
                        double weeklyMean = weeklyMeans.get(weekNumber);
                        double weeklyVariance = weeklyVariances.get(weekNumber);
                        double casesPerCapita = week.getCasesPerCapita();

                        double weeklyZScore = determineZScore(weeklyMean, weeklyVariance, casesPerCapita);
                        zScoreSum += weeklyZScore;
                    }

                    double averageZScore = zScoreSum / totalNumWeeks;
                    resultList.add(curCountyNumber + "," + averageZScore + "\n");
                }
            }

            // Check if the file already exists, if so delete so we can create a fresh one
            File fileName = new File("RawDataToUsableDataConverter/src/CovidSeverityIndices.csv");
            if (fileName.exists()) {
                fileName.delete();
            }

            // Now that we have all the severityIndices, we can create the CovidSeverityIndices.csv file to be used next step
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
     * Returns a list of integers representing the weekly cases / capita for the particular countyNumber
     * @param countyNumber The county FIPS code that you would like to find the weekly cases / capita
     * @return A List<Integer> representing weekly cases / capita
     */
    public static List<WeeklyDataPoint> getWeeklyCasesPerCapitaList(int countyNumber) {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader("RawDataToUsableDataConverter/src/TimeSeries/" + countyNumber + ".csv"));

            List<WeeklyDataPoint> weeklyCaseDataPoints = new ArrayList<>();
            double curCountyPopulation = (double) countyPopulations.get(countyNumber);

            int weekCounter = 1;
            int singleWeekCounter = 0;
            double singleWeekSum = 0;

            String curLine = "";
            while ((curLine = csvReader.readLine()) != null) {
                String[] data = curLine.split(",");

                int curDay = Integer.parseInt(data[0]);
                int countyNum = Integer.parseInt(data[1]);
                int deltaCases = Integer.parseInt(data[2]);
                int totalCases = Integer.parseInt(data[3]);
                int deltaDeaths = Integer.parseInt(data[4]);
                int totalDeaths = Integer.parseInt(data[5]);

                singleWeekSum += deltaCases;
                singleWeekCounter += 1;

                if (singleWeekCounter == 7) {
                    WeeklyDataPoint newDataPoint = new WeeklyDataPoint();
                    newDataPoint.setCountyNumber(countyNumber);
                    newDataPoint.setCountyPopulation((int) curCountyPopulation);
                    newDataPoint.setCasesPerCapita(singleWeekSum / curCountyPopulation);
                    newDataPoint.setWeekNumber(weekCounter);

                    weeklyCaseDataPoints.add(newDataPoint);

                    weekCounter += 1;
                    singleWeekSum = 0;
                    singleWeekCounter = 0;
                }
            }
            csvReader.close();

            if (singleWeekCounter != 0) {
                WeeklyDataPoint newDataPoint = new WeeklyDataPoint();
                newDataPoint.setCountyNumber(countyNumber);
                newDataPoint.setCountyPopulation((int) curCountyPopulation);
                newDataPoint.setCasesPerCapita(singleWeekSum / curCountyPopulation);
                newDataPoint.setWeekNumber(weekCounter);
            }

            return weeklyCaseDataPoints;
        } catch (NullPointerException | IOException d) {
            return null;
        }
    }

    public static double determineZScore(double mean, double variance, double value) {
        double numerator = value - mean;
        double standardDeviation = Math.sqrt(variance);
        return (numerator / standardDeviation);
    }

    private static void tearDownTempDirectoryStructure(Set<Integer> countyNumbers) {
        try {
            for (Integer county : countyNumbers) {
                File currentFile = new File("RawDataToUsableDataConverter/src/TimeSeries/" + county + ".csv");
                currentFile.delete();
            }
            File directory = new File("RawDataToUsableDataConverter/src/TimeSeries");
            directory.delete();
        } catch (Exception e) {

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

    public static void combineFeatureVectorAndResponseData() {
        combineFeaturesWithSeverityIndex();
        deleteIntermediateFiles();
    }

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

            File fileName = new File("RawDataToUsableDataConverter/src/FinalCombinedFeaturesWithSeverity.csv");
            if (fileName.exists()) {
                fileName.delete();
            }

            FileWriter csvWriter = new FileWriter("RawDataToUsableDataConverter/src/FinalCombinedFeaturesWithSeverity.csv");

            for (Integer countyNumber: countyNumberList) {
                double curCountySeverityIndex = countyCovidSeverityMap.get(countyNumber);
                String curCountyFeatureVector = countyDemographicsMap.get(countyNumber);
                String modifiedCurCountyFeatureVector = curCountyFeatureVector.replaceFirst("[0-9]+;", "");

                if (curCountyFeatureVector != null) {
                    csvWriter.append(String.valueOf(curCountySeverityIndex));
                    csvWriter.append(";");
                    //csvWriter.append(curCountyFeatureVector);
                    csvWriter.append(modifiedCurCountyFeatureVector);
                    csvWriter.append("\n");
                }

            }
            csvWriter.flush();
            csvWriter.close();

        } catch (IOException d) {
            d.printStackTrace();
        }
    }

    private static void deleteIntermediateFiles() {
        try {
            File intermediateFile = new File("RawDataToUsableDataConverter/src/CovidFileWithModifiedDates.csv");
            intermediateFile.delete();
        } catch (Exception e) {

        }
    }

    /**
     * End feature / response combination methods
     *******************************************************
     */

}
