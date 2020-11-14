import java.util.Objects;

public class WeeklyDataPoint {

    private int countyNumber;
    private int countyPopulation;
    private int weekNumber;
    private double casesPerCapita;

    public WeeklyDataPoint() {
        this.countyNumber = 0;
        this.countyPopulation = 0;
        this.weekNumber = 0;
        this.casesPerCapita = 0;
    }

    public WeeklyDataPoint(int countyNumber, int countyPopulation, int weekNumber, double casesPerCapita) {
        this.countyNumber = countyNumber;
        this.countyPopulation = countyPopulation;
        this.weekNumber = weekNumber;
        this.casesPerCapita = casesPerCapita;
    }

    public int getCountyNumber() {
        return countyNumber;
    }

    public void setCountyNumber(int countyNumber) {
        this.countyNumber = countyNumber;
    }

    public int getCountyPopulation() {
        return countyPopulation;
    }

    public void setCountyPopulation(int countyPopulation) {
        this.countyPopulation = countyPopulation;
    }

    public int getWeekNumber() {
        return weekNumber;
    }

    public void setWeekNumber(int weekNumber) {
        this.weekNumber = weekNumber;
    }

    public double getCasesPerCapita() {
        return casesPerCapita;
    }

    public void setCasesPerCapita(double casesPerCapita) {
        this.casesPerCapita = casesPerCapita;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WeeklyDataPoint that = (WeeklyDataPoint) o;
        return countyNumber == that.countyNumber &&
                countyPopulation == that.countyPopulation &&
                weekNumber == that.weekNumber &&
                Double.compare(that.casesPerCapita, casesPerCapita) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(countyNumber, countyPopulation, weekNumber, casesPerCapita);
    }

    @Override
    public String toString() {
        return "WeeklyDataPoint{" +
                "countyNumber=" + countyNumber +
                ", countyPopulation=" + countyPopulation +
                ", weekNumber=" + weekNumber +
                ", casesPerCapita=" + casesPerCapita +
                '}';
    }
}
