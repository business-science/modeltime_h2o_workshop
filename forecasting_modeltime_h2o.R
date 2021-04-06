# BUSINESS SCIENCE LEARNING LABS ----
# MODELTIME H2O WORKSHOP ----
# **** ----

# BUSINESS OBJECTIVE ----
# - Forecast intermittent demand 
# - Predict next 52-WEEKS
# **** ----


# LIBRARIES ----

library(tidymodels)
library(modeltime.h2o)
library(tidyverse)
library(timetk)

# DATA -----

walmart_sales_weekly

data_tbl <- walmart_sales_weekly %>%
    select(id, Date, Weekly_Sales)

# * Time Plot ----
data_tbl %>% 
    group_by(id) %>% 
    plot_time_series(
        .date_var    = Date,
        .value       = Weekly_Sales,
        .facet_ncol  = 2,
        .smooth      = TRUE,
        .smooth_period = "2 quarters",
        .interactive = TRUE
    )

# * Seasonality Plot ----
ids <- unique(data_tbl$id)

data_tbl %>% 
    filter(id == ids[2]) %>%
    plot_seasonal_diagnostics(
        .date_var    = Date,
        .value       = log(Weekly_Sales)
    )

# TRAIN / TEST SPLITS ---- 

FORECAST_HORIZON <- 52

splits <- time_series_split(data_tbl, assess = FORECAST_HORIZON, cumulative = TRUE)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(Date, Weekly_Sales)

# PREPROCESSING ----

recipe_spec <- recipe(Weekly_Sales ~ ., data = training(splits)) %>%
    step_timeseries_signature(Date) %>%
    step_normalize(Date_index.num, starts_with("Date_year")) 

recipe_spec %>% prep() %>% juice() %>% glimpse()

# MODELING ----

# Initialize H2O
h2o.init(
    nthreads = -1,
    ip       = 'localhost',
    port     = 54321
)

# Optional (Turn off progress)
h2o.no_progress()

# * Model Specification ----

model_spec_h2o <- automl_reg(mode = 'regression') %>%
    set_engine(
        engine                     = 'h2o',
        max_runtime_secs           = 30, 
        max_runtime_secs_per_model = 10,
        max_models                 = 30,
        nfolds                     = 5,
        exclude_algos              = c("DeepLearning"),
        verbosity                  = NULL,
        seed                       = 786
    ) 

model_spec_h2o

# * Fitting ----
#   - This step will take some time depending on your Model Specification selections

wflw_fit_h2o <- workflow() %>%
    add_model(model_spec_h2o) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_fit_h2o

# H2O Model Properties ----

wflw_fit_h2o %>% automl_leaderboard()

# Saving / Loading Models ----

wflw_fit_h2o %>%
    automl_update_model('XGBoost_grid__1_AutoML_20210406_140637_model_2') %>%
    save_h2o_model(path = 'h2o_models/XGBoost_grid__1_AutoML_20210406_140637_model_2')

load_h2o_model("h2o_models/XGBoost_grid__1_AutoML_20210406_140637_model_2/")

# FORECASTING ----

# * Modeltime Table ----
modeltime_tbl <- modeltime_table(
    wflw_fit_h2o,
    wflw_fit_h2o %>%
        automl_update_model('XGBoost_grid__1_AutoML_20210406_140637_model_2')
) 

modeltime_tbl

# * Calibrate ----

calibration_tbl <- modeltime_tbl %>%
    modeltime_calibrate(testing(splits)) 

# * Forecasting ----

calibration_tbl %>%
    modeltime_forecast(
        new_data    = testing(splits),
        actual_data = data_tbl,
        keep_data   = TRUE
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .facet_ncol  = 2, 
        .interactive = TRUE
    )

# * Refitting ----
#   - Working with Erin LeDell to provide option
#     for retraining specific models

refit_tbl <- calibration_tbl %>%
    modeltime_refit(data_tbl)

# * Future Forecast ----

future_tbl <- testing(splits) %>%
    group_by(id) %>%
    future_frame(Date, .length_out = 52) %>%
    ungroup()

refit_tbl %>%
    modeltime_forecast(
        new_data    = future_tbl,
        actual_data = data_tbl,
        keep_data   = TRUE
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .facet_ncol  = 2, 
        .interactive = TRUE
    )

# LEARNING MORE ----
# * HIGH-PERFORMANCE TIME SERIES COURSE ----



