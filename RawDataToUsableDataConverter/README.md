# RawDataToUsableDataConverter
A java program that works to transform raw data files into an aggregate feature vector / response matrix that can be used in further ML algorithms.

The program requires 4 files to work:
1.StartingFiles/CensusCountyPopulations
    * The predicted population data of each county as of 2019 per the US Government census obtained from
    https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/

2. StartingFiles/CovidFileWithDates.csv 
    * The raw county COVID data obtained from 
    https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
    that is used to derive the severity index of the COVID response for each county

3. StartingFiles/RawCountyDemographics.csv
    * The raw demographics data obtained from 
    [public.opendatasoft.com](https://public.opendatasoft.com/explore/dataset/usa-2016-presidential-election-by-county/table/?disjunctive.state&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiQVZHIiwieUF4aXMiOiJjYSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNFOTFEMEUifSx7InR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiQVZHIiwieUF4aXMiOiJkZW0xNl9mcmFjIiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWUsImNvbG9yIjoiIzIzMjA2NiJ9XSwieEF4aXMiOiJzdGF0ZSIsIm1heHBvaW50cyI6MjAwLCJzb3J0IjoiIiwiY29uZmlnIjp7ImRhdGFzZXQiOiJ1c2EtMjAxNi1wcmVzaWRlbnRpYWwtZWxlY3Rpb24tYnktY291bnR5Iiwib3B0aW9ucyI6eyJkaXNqdW5jdGl2ZS5zdGF0ZSI6dHJ1ZX19fV0sInRpbWVzY2FsZSI6IiIsImRpc3BsYXlMZWdlbmQiOnRydWUsImFsaWduTW9udGgiOnRydWV9)
    that is used to build the final feature vectors for each county.
    
4. StartingFiles/FeaturesToKeep.csv
    * A list (1 index per line) of all feature indices from the RawCountyDemographics.csv file
    that you want to keep for your final resulting feature matrix
    
    
After processing these files, the program will produce a number of intermediary files, as well as one named
* FinalCombinedFeaturesWithSeverity.csv 

This file will be what gets carried on to the other steps of our project to perform various ML algorithms on it. Think of this file
as the equivalent of what we begin with when looking at the Titanic Data sets.
