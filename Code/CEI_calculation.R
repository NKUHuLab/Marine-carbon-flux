library(readxl)
library(climdex.pcic)
library(PCICt)
library(zoo)
library(dplyr)
library(tidyr)
library(lubridate)
library(openxlsx)
source("Climpact.ETSCI_functions_all.R")
input_file_path <- "Sample_data_CEI.xlsx"
output_file_path <- "CEI_results.xlsx"
raw_data <- read_excel(input_file_path)
long_data <- raw_data %>%
  pivot_longer(
    cols = -Var,
    names_to = "date_str",
    values_to = "value"
  ) %>%
  extract(Var, into = c("grid_id", "variable"), regex = "(.*)-(prcp|tmax|tmin)$") %>%
  pivot_wider(
    names_from = variable,
    values_from = value
  ) %>%
  mutate(
    date = ymd(date_str),
    year = year(date),
    month = month(date),
    day = day(date)
  ) %>%
  select(grid_id, date, year, month, day, prcp, tmax, tmin)
unique_grid_ids <- unique(long_data$grid_id)
list_of_data <- split(long_data, factor(long_data$grid_id, levels = unique_grid_ids))
rm(long_data)
gc()
process_chunk <- function(daily_data) {
  if (is.null(daily_data) || nrow(daily_data) == 0) {
    return(NULL)
  }
  current_grid_id <- daily_data$grid_id[1]
  pcict_dates <- as.PCICt(as.character(daily_data$date), cal = "gregorian")
  base_period_percentiles <- c(1981, 2010)
  climdex_input_object_0 <- climdex.pcic::climdexInput.raw(
    tmax = daily_data$tmax,
    tmin = daily_data$tmin,
    prec = daily_data$prcp,
    tmax.dates = pcict_dates,
    tmin.dates = pcict_dates,
    prec.dates = pcict_dates,
    base.range = base_period_percentiles
  )
  base_period <- c(min(daily_data$year), max(daily_data$year))
  climdex_input_object <- climdex.pcic::climdexInput.raw(
    tmax = daily_data$tmax,
    tmin = daily_data$tmin,
    prec = daily_data$prcp,
    tmax.dates = pcict_dates,
    tmin.dates = pcict_dates,
    prec.dates = pcict_dates,
    base.range = base_period
  )
  monthly_cdd <- climdex.cdd(daily_data)
  monthly_prcptot <- climdex.prcptot(climdex_input_object, freq = "monthly")
  monthly_r50mm <- climdex.rnnmm(climdex_input_object, threshold = 50, freq = "monthly")
  monthly_r99p <- climdex.r99p(climdex_input_object_0)
  monthly_rx1day <- climdex.rxnday(daily_data, n = 1)
  monthly_su <- climdex.su(climdex_input_object, freq = "monthly")
  monthly_tr <- climdex.tr(climdex_input_object, freq = "monthly")
  time_labels <- names(monthly_prcptot)
  formatted_time <- gsub("-", "", time_labels)
  result_df <- data.frame(
    Time = formatted_time,
    CDD = as.numeric(monthly_cdd),
    PRCPTOT = as.numeric(monthly_prcptot),
    R50mm = as.numeric(monthly_r50mm),
    R99p = as.numeric(monthly_r99p),
    Rx1day = as.numeric(monthly_rx1day),
    SU = as.numeric(monthly_su),
    TR = as.numeric(monthly_tr)
  )
  result_long <- result_df %>%
    pivot_longer(
      cols = -Time,
      names_to = "index_name",
      values_to = "value"
    ) %>%
    mutate(
      Var = paste0(current_grid_id, "-", index_name)
    ) %>%
    select(Var, Time, value)
  return(result_long)
}
all_results_list <- lapply(list_of_data, process_chunk)
final_results_long <- bind_rows(all_results_list)
final_results_wide <- final_results_long %>%
  pivot_wider(
    names_from = Time,
    values_from = value
  ) %>%
  select(Var, everything())
write.xlsx(final_results_wide, output_file_path)