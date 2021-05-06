beach_locations <- read.csv("~/projects/portfolio/IndependentResearch/ecoli/data/Beach_Water_and_Weather_Sensor_Locations.csv")
beach_lab_data <- read.csv("~/projects/portfolio/IndependentResearch/ecoli/data/Beach_Lab_Data.csv")
beach_weather_data <- read.csv("~/projects/portfolio/IndependentResearch/ecoli/data/Beach_Weather_Stations_Automated_Sensors.csv")
beach_water_data <- read.csv("~/projects/portfolio/IndependentResearch/ecoli/data/Beach_Water_Quality_Automated_Sensors.csv")

require(tidyverse)

dist.haversine <- function(lat1, lon1, lat2, lon2) {
  # radius of earth in km
  R <- 6371
  
  # convert to radians for this to work
  toRad <- pi/180
  lat1 <- lat1 * toRad
  lat2 <- lat2 * toRad
  lon1 <- lon1 * toRad
  lon2 <- lon2 * toRad
  a <- sin((lat2 - lat1) / 2)^2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  return (R * c)
}

beach_weather_data <-
  beach_weather_data %>% 
  mutate(
    Measurement.ID.std = str_remove(Measurement.ID, "WeatherStation")
  )

beach_water_data <-
  beach_water_data %>% 
  mutate(
    Measurement.ID.std = str_remove(Measurement.ID, "Beach")
  )

# treat the grain of this dataset as a Beach even though the sensors are in technically different locations.
# e.g. 63rd Street Beach is a different coordinate than 63rd Street Weather Station
beach_sensor_data <-
    beach_water_data %>% 
    # The beaches represented in the weather dataset is a subset of those in the water dataset, hence the left join
    left_join(beach_weather_data, by = "Measurement.ID.std", suffix = c(".weather", ".water"))

# clean lab data
data <- 
  beach_lab_data %>% 
  # if a measurement cannot be tied to a beach, then we're not interested.
  # coincidentally, the measurements filtered out also don't have Lat/Lon data so the measurements don't do much good anyway.
  filter(!is.na(Beach)) %>% 
  mutate(
    Beach = paste0(
        str_replace_all(Beach, 
                    c("Marion Mahony Griffin \\(Jarvis\\)" = "Marion Mahony Griffin",
                      "Margaret T Burroughs \\(31st\\)" = "Margaret T Burroughs")
        ),
      " Beach"),
    Timestamp.key = coalesce(DNA.Sample.Timestamp, Culture.Sample.1.Timestamp, Culture.Sample.2.Timestamp)
  ) %>% 
  # since beach_sensor_data joins on Timestamps, picking either measurement timestamp works for this join.
  # we'll pick water though since it was the left set in the join above
  left_join(beach_sensor_data, by = c("Beach" = "Beach.Name", "Timestamp.key" = "Measurement.Timestamp.water"))

# Items to fix
# 1. Correct beach identification for 31st st and Jarvis Beaches (check)
# 2. impute missing temperature and water with nearest beach (if reasonable)
beach_locs <- data %>% select(Beach, Latitude, Longitude) %>% distinct %>% filter(!is.na(Latitude))
beach.list <- beach_locs$Beach
beach_combos <- t(combn(beach.list, m = 2)) %>% as.data.frame()
colnames(beach_combos) <- c("A", "B")

# Calculate closest beach using haversine distance formula
a <- 
  beach_combos %>% 
  inner_join(beach_locs, by = c("A" = "Beach")) %>% 
    rename(Latitude.A = Latitude, Longitude.A = Longitude) %>% 
  inner_join(beach_locs, by = c("B" = "Beach")) %>% 
    rename(Latitude.B = Latitude, Longitude.B = Longitude) %>%
  mutate(
    dist = dist.haversine(Latitude.A, Longitude.A, Latitude.B, Longitude.B)
  )

data <- 
  a %>% 
    group_by(A) %>% 
    summarise(min = min(dist)) %>% 
    inner_join(a, by = c("A" = "A", "min" = "dist")) %>% 
    # bring distance just in case we want it to include it for penalizing the algorithm
    select(Beach = A, Closest.Beach = B, DistanceToClosestBeach = min) %>% 
    inner_join(data, by = "Beach")