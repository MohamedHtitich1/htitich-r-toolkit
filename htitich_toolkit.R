#' ============================================================================
#' HTITICH R TOOLKIT
#' A Comprehensive Library for Sustainable Development Measurement & Analysis
#' ============================================================================
#' 
#' Author: Mohamed Htitich, PhD
#' Affiliation: Social Progress Imperative (2021-2025)
#'              Palacký University Olomouc
#' 
#' Description: This toolkit consolidates R functions and workflows developed
#' during research on sustainable development measurement, including:
#'   - Composite index construction (SPI, JTS, CESP, CPI)
#'   - Entropy weighting methods
#'   - Convergence analysis (Phillips-Sul club convergence)
#'   - Panel data econometrics
#'   - Machine learning for survey data (Random Forest, Classification Trees)
#'   - Multivariate ordination (NMDS)
#'   - Spatial visualization (Tableau/GeoJSON, Eurostat NUTS mapping)
#'   - Data imputation (cubic smoothing, missForest)
#'
#' ============================================================================

# =============================================================================
# SECTION 1: REQUIRED PACKAGES
# =============================================================================

#' Install and load required packages
#' @param packages Character vector of package names
load_packages <- function(packages = NULL) {
  if (is.null(packages)) {
    packages <- c(
      # Data manipulation
      "dplyr", "tidyr", "data.table", "stringr", "purrr", "readr",
      # Data import/export
      "readxl", "writexl", "haven", "openxlsx",
      # Visualization
      "ggplot2", "viridis", "ggrepel", "gghighlight", "ggpattern", "cowplot",
      # Panel data & econometrics
      "plm", "AER", "stargazer",
      # Composite indices
      "Compind", "creditmodel",
      # Convergence analysis
      "ConvergenceClubs",
      # Machine learning
      "randomForest", "partykit", "ggparty", "missForest",
      # Multivariate analysis
      "vegan", "indicspecies",
      # Spatial/mapping
      "sf", "tmap", "eurostat", "maps",
      # Statistical
      "psych", "REAT", "segmented",
      # Imputation
      "mgcv", "zoo",
      # Tables
      "gt", "gtExtras", "DT"
    )
  }
  
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message(sprintf("Installing package: %s", pkg))
      install.packages(pkg, dependencies = TRUE)
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
  message("All packages loaded successfully.")
}

# =============================================================================
# SECTION 2: UTILITY FUNCTIONS
# =============================================================================

#' Repeat last non-NA value forward (for filling missing values in time series)
#' @param x Numeric or character vector
#' @return Vector with NA values filled forward
#' @examples
#' repeat_before(c(NA, 1, NA, 3, NA, NA, 5))
repeat_before <- function(x) {
  ind <- which(!is.na(x))
  if (is.na(x[1])) ind <- c(1, ind)
  rep(x[ind], times = diff(c(ind, length(x) + 1)))
}

#' Rescale vector to 0-100 range
#' @param x Numeric vector
#' @return Numeric vector scaled to 0-100
rescale_0_100 <- function(x) {
  r <- range(x, na.rm = TRUE)
  if (!is.finite(r[1]) || !is.finite(r[2]) || r[1] == r[2]) {
    return(rep(NA_real_, length(x)))
  }
  (x - r[1]) / (r[2] - r[1]) * 100
}

#' Safe logarithm (returns NA for non-positive values)
#' @param x Numeric vector
#' @param base Logarithm base (default 10)
#' @return Numeric vector of log values
safe_log <- function(x, base = 10) {
  ifelse(is.na(x) | x <= 0, NA_real_, log(x, base = base))
}

#' Calculate geometric mean
#' @param x Numeric vector
#' @param na.rm Logical, remove NA values
#' @return Geometric mean
geometric_mean <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  if (length(x) == 0 || any(x <= 0)) return(NA_real_)
  exp(mean(log(x)))
}

#' Filter EU27 countries from dataset with NUTS geo codes
#' @param df Data frame with 'geo' column
#' @param extra Additional country codes to keep
#' @return Filtered data frame
keep_eu27 <- function(df, extra = c("NO", "IS")) {
  eu27 <- c("AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL",
            "HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK",
            "SI","ES","SE")
  keep <- union(eu27, extra)
  df %>%
    mutate(ctry = substr(geo, 1, 2),
           ctry = ifelse(ctry == "GR", "EL", ctry)) %>%
    filter(ctry %in% keep) %>%
    select(-ctry)
}

# =============================================================================
# SECTION 3: DATA IMPUTATION METHODS
# =============================================================================

#' Cubic smoothing spline imputation for time series
#' @param data Data frame with country, spiyear, and variable columns
#' @param var_name Name of variable to impute
#' @param country_col Name of country grouping column
#' @param year_col Name of year column
#' @return Data frame with imputed values and prediction indicator
cubic_smoothing_impute <- function(data, var_name, country_col = "country", 
                                    year_col = "spiyear") {
  require(mgcv)
  
  predgam <- function(data, model, var = "pred", type = "response") {
    data[[var]] <- predict(model, data, type = "response")
    data
  }
  
  data_gam <- data %>%
    group_by(!!sym(country_col)) %>%
    filter(any(is.na(!!sym(var_name))) & sum(!is.na(!!sym(var_name))) > 2) %>%
    filter(!all(is.na(!!sym(var_name)))) %>%
    nest() %>%
    mutate(
      mdl = map(data, ~ gam(
        as.formula(paste(var_name, "~ s(", year_col, ", bs = 'cr', k = sum(!is.na(.$", var_name, ")), fx = FALSE)")),
        data = .
      )),
      data = map2(data, mdl, predgam)
    ) %>%
    select(-mdl) %>%
    unnest(cols = c(data))
  
  data_gam
}

#' Linear interpolation with constant endpoint extension
#' @param y Numeric vector
#' @return List with interpolated values and flag for was_na
interp_const_ends <- function(y) {
  n <- length(y)
  idx <- which(!is.na(y))
  was_na <- is.na(y)
  
  if (!length(idx)) return(list(value = y, flag = as.integer(was_na)))
  if (length(idx) == 1) return(list(value = rep(y[idx], n), flag = as.integer(was_na)))
  
  v <- approx(x = idx, y = y[idx], xout = seq_len(n), method = "linear", rule = 2)$y
  list(value = v, flag = as.integer(was_na))
}

# =============================================================================
# SECTION 4: COMPOSITE INDEX CONSTRUCTION
# =============================================================================

#' Calculate entropy weights for indicators
#' @param data Data frame with indicators
#' @param neg_vars Names of negative polarity variables
#' @param pos_vars Names of positive polarity variables
#' @return Data frame with indicator weights
entropy_weights <- function(data, neg_vars = NULL, pos_vars = NULL) {
  require(creditmodel)
  entropy_weight(data, neg_vars = neg_vars, pos_vars = pos_vars)
}

#' Calculate Carbon Efficiency of Social Progress (CESP)
#' @param spi_data Data frame with SPI scores
#' @param co2_data Data frame with CO2 per capita
#' @param spi_col Name of SPI score column
#' @param co2_col Name of CO2 per capita column
#' @param country_col Name of country column
#' @param year_col Name of year column
#' @return Data frame with CESP scores
calculate_cesp <- function(spi_data, co2_data = NULL, 
                           spi_col = "score_spi", co2_col = "co2_percap",
                           country_col = "country", year_col = "spiyear") {
  
  # Merge if separate data frames
  if (!is.null(co2_data)) {
    data <- spi_data %>%
      left_join(co2_data, by = c(country_col, year_col))
  } else {
    data <- spi_data
  }
  
  # Calculate SD and mean for equalization constant
  sd_mean <- data %>%
    ungroup() %>%
    summarise(
      spi_mean = mean(!!sym(spi_col), na.rm = TRUE),
      spi_sd = sd(!!sym(spi_col), na.rm = TRUE),
      co2_mean = mean(!!sym(co2_col), na.rm = TRUE),
      co2_sd = sd(!!sym(co2_col), na.rm = TRUE)
    ) %>%
    mutate(const_cesp = ((co2_sd * spi_mean) / spi_sd) - co2_mean)
  
  # Define utopia/dystopia reference points
  utopia_cesp <- 0.1758
  dystopia_cesp <- 1.5
  
  # Calculate CESP
  data %>%
    mutate(
      adj_co2 = !!sym(co2_col) + sd_mean$const_cesp,
      cesp_raw = adj_co2 / !!sym(spi_col)
    ) %>%
    mutate(
      score_cesp = (dystopia_cesp - cesp_raw) / (dystopia_cesp - utopia_cesp) * 100,
      score_cesp = pmax(0, pmin(100, score_cesp))  # Bound to 0-100
    )
}

#' Calculate Just Transition Score (JTS) combining CO2 and material footprint
#' @param data Data frame with SPI, CO2, and material footprint
#' @param spi_col Name of SPI column
#' @param co2_col Name of CO2 per capita column  
#' @param matfoot_col Name of material footprint column
#' @return Data frame with JTS scores
calculate_jts <- function(data, spi_col = "score_spi", 
                          co2_col = "co2_percap", matfoot_col = "matfoot_percap") {
  
  # Calculate CESP component
  data <- calculate_cesp(data, spi_col = spi_col, co2_col = co2_col)
  
  # Calculate material footprint component (MFSP)
  sd_mean_mf <- data %>%
    ungroup() %>%
    summarise(
      spi_mean = mean(!!sym(spi_col), na.rm = TRUE),
      spi_sd = sd(!!sym(spi_col), na.rm = TRUE),
      mf_mean = mean(!!sym(matfoot_col), na.rm = TRUE),
      mf_sd = sd(!!sym(matfoot_col), na.rm = TRUE)
    ) %>%
    mutate(const_mfsp = ((mf_sd * spi_mean) / spi_sd) - mf_mean)
  
  data %>%
    mutate(
      adj_mf = !!sym(matfoot_col) + sd_mean_mf$const_mfsp,
      mfsp_raw = adj_mf / !!sym(spi_col),
      # Use same min-max normalization
      score_mfsp = rescale_0_100(-mfsp_raw),  # Negative because lower is better
      # JTS is arithmetic mean of CESP and MFSP
      score_jts = (score_cesp + score_mfsp) / 2
    )
}

#' Mazziotta-Pareto Index calculation
#' @param data Data frame with indicators
#' @param indic_col Column indices of indicators
#' @param polarity Vector of "POS" or "NEG" for each indicator
#' @param time Time variable
#' @param penalty "POS" for positive penalty on imbalance
#' @return Composite index values
calculate_mpi <- function(data, indic_col, polarity, time = NULL, penalty = "POS") {
  require(Compind)
  ci_wampi_original(data, indic_col = indic_col, polarity = polarity, 
                    time = time, penalty = penalty)
}

# =============================================================================
# SECTION 5: CONVERGENCE ANALYSIS
# =============================================================================

#' Phillips-Sul Club Convergence Analysis
#' @param data Wide-format data with countries as rows, years as columns
#' @param unit_col Column index for country names
#' @param data_cols Column indices for year data
#' @param cstar Critical value for club formation (default 0)
#' @return List with convergence results
phillips_sul_convergence <- function(data, unit_col = 1, data_cols = 4:34, 
                                     cstar = 0, time_trim = 1/3, 
                                     HACmethod = "FQSB") {
  require(ConvergenceClubs)
  
  # Compute relative transition paths
  H <- computeH(data[, data_cols], quantity = "both")
  
  # Test for global convergence
  global_test <- estimateMod(H$H, time_trim = time_trim, HACmethod = HACmethod)
  
  # Find convergence clubs
  clubs <- findClubs(as.data.frame(data), dataCols = data_cols, 
                     unit_names = unit_col, refCol = max(data_cols),
                     cstar = cstar, HACmethod = HACmethod)
  
  list(
    H = H,
    global_test = global_test,
    clubs = clubs,
    transition_paths = H$h
  )
}

#' Extract club membership from convergence results
#' @param clubs Output from findClubs
#' @return Data frame with country-club mapping
extract_club_membership <- function(clubs) {
  club_list <- list()
  i <- 1
  while (!is.null(clubs[[paste0("club", i)]])) {
    club_list[[i]] <- data.frame(
      country = clubs[[paste0("club", i)]][["unit_names"]],
      club = paste("Club", i),
      stringsAsFactors = FALSE
    )
    i <- i + 1
  }
  bind_rows(club_list)
}

# =============================================================================
# SECTION 6: SEGMENTED REGRESSION ANALYSIS
# =============================================================================

#' Fit segmented regression with automatic breakpoint detection
#' @param data Data frame
#' @param formula Model formula (y ~ x)
#' @param seg_var Variable for breakpoints
#' @param n_breaks Expected number of breakpoints
#' @param type Selection criterion ("bic" or "aic")
#' @return Segmented model object
fit_segmented <- function(data, formula, seg_var, n_breaks = 2, 
                          type = "bic", n_boot = 1000) {
  require(segmented)
  
  # Fit initial linear model
  lm_model <- lm(formula, data = data)
  
  # Select breakpoints using BIC/AIC
  sel_breaks <- selgmented(lm_model, type = type, plot.ic = FALSE, 
                           Kmax = 10, stop.if = 100)
  
  # Fit final segmented model
  seg_formula <- as.formula(paste("~", seg_var))
  initial_psi <- quantile(data[[seg_var]], probs = seq(0.2, 0.8, length.out = n_breaks), 
                          na.rm = TRUE)
  
  seg_model <- segmented(lm_model, seg.Z = seg_formula,
                         psi = list(setNames(as.numeric(initial_psi), seg_var)),
                         type = type,
                         control = seg.control(n.boot = n_boot, it.max = 100, 
                                               conv.psi = TRUE, alpha = 0.01))
  
  seg_model
}

#' Extract segment parameters from segmented regression
#' @param seg_model Segmented model object
#' @return Data frame with intercepts and slopes for each segment
extract_segment_params <- function(seg_model) {
  coefs <- coef(seg_model)
  breakpoints <- seg_model$psi[, "Est."]
  
  # First segment
  b0 <- coefs[1]
  b1 <- coefs[2]
  
  params <- data.frame(
    segment = 1,
    intercept = b0,
    slope = b1,
    start = -Inf,
    end = breakpoints[1]
  )
  
  # Additional segments
  for (i in seq_along(breakpoints)) {
    slope_change <- coefs[2 + i]
    prev_slope <- params$slope[i]
    new_slope <- prev_slope + slope_change
    prev_intercept <- params$intercept[i]
    
    # Calculate new intercept at breakpoint
    new_intercept <- prev_intercept + prev_slope * breakpoints[i] - new_slope * breakpoints[i]
    
    end_val <- if (i < length(breakpoints)) breakpoints[i + 1] else Inf
    
    params <- rbind(params, data.frame(
      segment = i + 1,
      intercept = new_intercept,
      slope = new_slope,
      start = breakpoints[i],
      end = end_val
    ))
  }
  
  params
}

# =============================================================================
# SECTION 7: MACHINE LEARNING FOR SURVEY DATA
# =============================================================================

#' Random Forest variable importance analysis
#' @param data Data frame
#' @param response Name of response variable
#' @param predictors Names of predictor variables
#' @param ntree Number of trees
#' @return List with model and variable importance
rf_variable_importance <- function(data, response, predictors, ntree = 500) {
  require(randomForest)
  
  formula <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  
  rf_model <- randomForest(formula, data = data, ntree = ntree, importance = TRUE)
  
  varimp <- importance(rf_model)
  varimp_df <- data.frame(
    variable = rownames(varimp),
    importance = varimp[, 1],
    scaled_importance = rescale_0_100(varimp[, 1])
  ) %>%
    arrange(desc(importance))
  
  list(model = rf_model, importance = varimp_df)
}

#' Classification tree with visualization
#' @param data Data frame
#' @param response Name of response variable (factor)
#' @param predictors Names of predictor variables
#' @param max_depth Maximum tree depth
#' @return Conditional inference tree object
fit_ctree <- function(data, response, predictors, max_depth = 3) {
  require(partykit)
  
  formula <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  
  ctree(formula, data = data, 
        control = ctree_control(maxdepth = max_depth))
}

# =============================================================================
# SECTION 8: MULTIVARIATE ORDINATION (NMDS)
# =============================================================================

#' Non-Metric Multidimensional Scaling analysis
#' @param data Data frame with species/variable columns
#' @param species_cols Column indices for species/variables
#' @param group_col Name of grouping variable
#' @param distance Distance measure ("bray", "euclidean", etc.)
#' @param trymax Maximum iterations
#' @return List with NMDS results, site scores, and species scores
nmds_analysis <- function(data, species_cols, group_col = NULL, 
                          distance = "bray", trymax = 250) {
  require(vegan)
  
  # Calculate distance matrix
  dist_mat <- vegdist(data[, species_cols], method = distance)
  
  # Run NMDS
  nmds <- metaMDS(dist_mat, wascores = TRUE, trymax = trymax)
  sppscores(nmds) <- data[, species_cols]
  
  # Extract site scores
  site_scores <- as.data.frame(scores(nmds, display = "sites"))
  if (!is.null(group_col) && group_col %in% names(data)) {
    site_scores[[group_col]] <- data[[group_col]]
  }
  
  # Extract species scores
  species_scores <- as.data.frame(scores(nmds, display = "species"))
  species_scores$variable <- rownames(species_scores)
  
  # Calculate centroids if grouping variable provided
  centroids <- NULL
  if (!is.null(group_col) && group_col %in% names(site_scores)) {
    centroids <- aggregate(cbind(NMDS1, NMDS2) ~ get(group_col), 
                           data = site_scores, FUN = mean)
    names(centroids)[1] <- group_col
  }
  
  list(
    nmds = nmds,
    site_scores = site_scores,
    species_scores = species_scores,
    centroids = centroids,
    stress = nmds$stress
  )
}

#' ANOSIM test for group differences
#' @param data Data frame with species columns
#' @param species_cols Column indices
#' @param group_var Grouping variable
#' @param distance Distance measure
#' @param permutations Number of permutations
#' @return ANOSIM result
test_group_differences <- function(data, species_cols, group_var, 
                                   distance = "bray", permutations = 9999) {
  require(vegan)
  anosim(as.matrix(data[, species_cols]), data[[group_var]], 
         distance = distance, permutations = permutations)
}

# =============================================================================
# SECTION 9: EUROSTAT API AND SPATIAL FUNCTIONS
# =============================================================================

#' Fetch Eurostat data at specified NUTS level
#' @param code Eurostat dataset code
#' @param level NUTS level (0, 1, 2, or 3)
#' @param years Years to retrieve
#' @return Data frame with Eurostat data
get_nuts_level <- function(code, level = 2, years = NULL) {
  require(eurostat)
  require(stringr)
  
  len_for_level <- c(`0` = 2, `1` = 3, `2` = 4, `3` = 5)
  target_len <- len_for_level[as.character(level)]
  
  df <- get_eurostat(code, time_format = "raw", stringsAsFactors = FALSE, cache = TRUE)
  
  # Normalize time column
  if (!"time" %in% names(df)) {
    if ("TIME_PERIOD" %in% names(df)) {
      df <- rename(df, time = TIME_PERIOD)
    } else {
      stop("No time column detected in dataset: ", code)
    }
  }
  
  df <- df %>%
    mutate(time = as.integer(substr(as.character(time), 1, 4))) %>%
    filter(str_length(geo) == target_len)
  
  if (!is.null(years)) {
    df <- df %>% filter(!is.na(time) & time %in% years)
  }
  
  df
}

#' Build NUTS2 reference skeleton from Eurostat geospatial data
#' @param nuts_year NUTS classification year
#' @return Data frame with NUTS2 codes and parent codes
build_nuts2_reference <- function(nuts_year = 2024) {
  require(eurostat)
  require(sf)
  
  geodata <- get_eurostat_geospatial(
    nuts_level = 2, year = nuts_year, resolution = "60",
    cache = TRUE, update_cache = TRUE, output_class = "sf", crs = 4326
  )
  
  geodata %>%
    st_drop_geometry() %>%
    transmute(
      geo = NUTS_ID,
      nuts1 = substr(NUTS_ID, 1, 3),
      nuts0 = substr(NUTS_ID, 1, 2),
      name = coalesce(NAME_LATN, NUTS_NAME)
    )
}

#' Cascade data from NUTS0/NUTS1 to NUTS2 level
#' @param data Data frame with geo and year columns
#' @param vars Variables to cascade
#' @param nuts2_ref NUTS2 reference (from build_nuts2_reference)
#' @param years Years to include
#' @return Data frame with values at NUTS2 level
cascade_to_nuts2 <- function(data, vars, nuts2_ref = NULL, years = NULL) {
  
  if (is.null(nuts2_ref)) {
    nuts2_ref <- build_nuts2_reference()
  }
  
  if (is.null(years)) {
    years <- sort(unique(data$year))
  }
  
  # Create skeleton
  skeleton <- tidyr::expand_grid(nuts2_ref, tibble::tibble(year = years))
  
  cascade_var <- function(var) {
    d <- data %>% select(geo, year, !!sym(var))
    
    v2 <- d %>% filter(nchar(geo) == 4) %>% rename(val2 = !!sym(var))
    v1 <- d %>% filter(nchar(geo) == 3) %>% transmute(nuts1 = geo, year, val1 = !!sym(var))
    v0 <- d %>% filter(nchar(geo) == 2) %>% transmute(nuts0 = geo, year, val0 = !!sym(var))
    
    skeleton %>%
      left_join(v2, by = c("geo", "year")) %>%
      left_join(v1, by = c("nuts1", "year")) %>%
      left_join(v0, by = c("nuts0", "year")) %>%
      mutate(
        !!var := coalesce(val2, val1, val0),
        !!paste0("src_", var, "_level") := case_when(
          !is.na(val2) ~ 2L,
          is.na(val2) & !is.na(val1) ~ 1L,
          is.na(val2) & is.na(val1) & !is.na(val0) ~ 0L,
          TRUE ~ NA_integer_
        )
      ) %>%
      select(geo, year, !!sym(var), !!sym(paste0("src_", var, "_level")))
  }
  
  result <- Reduce(function(a, b) left_join(a, b, by = c("geo", "year")), 
                   lapply(vars, cascade_var))
  
  result %>%
    left_join(select(nuts2_ref, geo, nuts1, nuts0, name), by = "geo")
}

# =============================================================================
# SECTION 10: VISUALIZATION FUNCTIONS
# =============================================================================

#' Create world choropleth map
#' @param data Data frame with country codes and values
#' @param value_col Name of value column
#' @param code_col Name of country code column (ISO3)
#' @param title Map title
#' @param palette Viridis palette option
#' @return ggplot object
world_choropleth <- function(data, value_col, code_col = "spicountrycode",
                             title = "", palette = "G") {
  require(maps)
  require(viridis)
  
  world <- map_data("world") %>%
    mutate(spicountrycode = iso.alpha(region, n = 3))
  
  world_data <- world %>%
    left_join(data, by = setNames(code_col, "spicountrycode")) %>%
    filter(region != "Antarctica")
  
  ggplot(world_data, aes(x = long, y = lat, group = group, fill = !!sym(value_col))) +
    coord_map("moll") +
    geom_polygon(color = "black", size = 0.05) +
    scale_fill_viridis(option = palette, na.value = "grey90") +
    labs(title = title, fill = "") +
    theme_void() +
    theme(
      legend.position = "right",
      legend.key.height = unit(0.6, "cm"),
      legend.key.width = unit(0.5, "cm")
    )
}

#' Publication-ready theme for ggplot
#' @param base_size Base font size
#' @return ggplot theme
theme_publication <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(face = "bold", size = base_size - 1),
      legend.text = element_text(size = base_size - 2),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(size = base_size - 2),
      strip.text = element_text(face = "bold"),
      strip.background = element_blank(),
      panel.border = element_rect(linetype = "solid", fill = NA),
      axis.ticks = element_line()
    )
}

#' Create trajectory plot for time series data
#' @param data Data frame with time series
#' @param x_col Time variable
#' @param y_col Value variable
#' @param group_col Grouping variable
#' @param label_col Label for endpoints
#' @param highlight_years Years to highlight with points
#' @return ggplot object
trajectory_plot <- function(data, x_col, y_col, group_col, label_col = NULL,
                            highlight_years = NULL) {
  require(ggrepel)
  
  p <- ggplot(data, aes(x = !!sym(x_col), y = !!sym(y_col), 
                        group = !!sym(group_col), color = !!sym(group_col))) +
    geom_line(size = 0.7, alpha = 0.7)
  
  if (!is.null(highlight_years)) {
    p <- p + geom_point(
      data = data %>% filter(!!sym(x_col) %in% highlight_years),
      size = 2, shape = 21, fill = "white"
    )
  }
  
  if (!is.null(label_col)) {
    p <- p + geom_text_repel(
      data = data %>% filter(!!sym(x_col) == max(!!sym(x_col))),
      aes(label = !!sym(label_col)),
      size = 2.5, nudge_x = 0.5, direction = "y", hjust = 0
    )
  }
  
  p + theme_publication() + theme(legend.position = "none")
}

#' Create scatter plot with labeled points
#' @param data Data frame
#' @param x_col X variable
#' @param y_col Y variable
#' @param label_col Label variable
#' @param color_col Color variable (optional)
#' @param threshold_x X threshold line (optional)
#' @param threshold_y Y threshold line (optional)
#' @return ggplot object
scatter_labeled <- function(data, x_col, y_col, label_col, color_col = NULL,
                            threshold_x = NULL, threshold_y = NULL) {
  require(ggrepel)
  
  p <- ggplot(data, aes(x = !!sym(x_col), y = !!sym(y_col)))
  
  if (!is.null(color_col)) {
    p <- p + geom_point(aes(color = !!sym(color_col)), size = 3, alpha = 0.7) +
      geom_point(shape = 21, size = 3, color = "black")
  } else {
    p <- p + geom_point(size = 3, shape = 21, color = "black", fill = "dodgerblue")
  }
  
  p <- p + geom_text_repel(
    aes(label = !!sym(label_col)),
    size = 2.5, box.padding = 0.3, max.overlaps = 10
  )
  
  if (!is.null(threshold_x)) {
    p <- p + geom_vline(xintercept = threshold_x, linetype = "dashed", color = "red")
  }
  
  if (!is.null(threshold_y)) {
    p <- p + geom_hline(yintercept = threshold_y, linetype = "dashed", color = "red")
  }
  
  p + theme_publication()
}

# =============================================================================
# SECTION 11: EXPORT FUNCTIONS
# =============================================================================

#' Export spatial data for Tableau
#' @param data sf object with geometry
#' @param filepath Output path (.geojson)
#' @param crs Coordinate reference system (default WGS84)
export_for_tableau <- function(data, filepath, crs = 4326) {
  require(sf)
  
  data %>%
    st_transform(crs) %>%
    st_cast("MULTIPOLYGON", warn = FALSE) %>%
    st_write(filepath, delete_dsn = TRUE)
  
  message("Exported: ", filepath)
}

#' Create summary statistics table
#' @param data Data frame
#' @param vars Variables to summarize
#' @param group_var Grouping variable (optional)
#' @return gt table object
summary_table <- function(data, vars, group_var = NULL) {
  require(gt)
  
  if (!is.null(group_var)) {
    summary_df <- data %>%
      group_by(!!sym(group_var)) %>%
      summarise(across(all_of(vars), 
                       list(mean = ~mean(.x, na.rm = TRUE),
                            sd = ~sd(.x, na.rm = TRUE),
                            n = ~sum(!is.na(.x)))),
                .groups = "drop")
  } else {
    summary_df <- data %>%
      summarise(across(all_of(vars),
                       list(mean = ~mean(.x, na.rm = TRUE),
                            sd = ~sd(.x, na.rm = TRUE),
                            n = ~sum(!is.na(.x)))))
  }
  
  summary_df %>%
    gt() %>%
    fmt_number(decimals = 2) %>%
    tab_header(title = "Summary Statistics")
}

# =============================================================================
# SECTION 12: EXAMPLE WORKFLOWS
# =============================================================================

#' Example: Calculate CESP for country panel data
#' @examples
#' \dontrun{
#' # Load data
#' spi_data <- read_xlsx("spi_scores.xlsx")
#' co2_data <- read_xlsx("co2_emissions.xlsx")
#' 
#' # Calculate CESP
#' results <- calculate_cesp(
#'   spi_data = spi_data,
#'   co2_data = co2_data,
#'   spi_col = "score_spi",
#'   co2_col = "co2_percap"
#' )
#' 
#' # Visualize
#' world_choropleth(
#'   results %>% filter(spiyear == 2020),
#'   value_col = "score_cesp",
#'   title = "Carbon Efficiency of Social Progress (2020)"
#' )
#' }

#' Example: Club Convergence Analysis
#' @examples
#' \dontrun{
#' # Prepare wide-format data
#' wide_data <- data %>%
#'   select(country, spiyear, score_cesp) %>%
#'   spread(spiyear, score_cesp)
#' 
#' # Run convergence analysis
#' conv_results <- phillips_sul_convergence(
#'   wide_data,
#'   unit_col = 1,
#'   data_cols = 2:32
#' )
#' 
#' # Extract clubs
#' clubs <- extract_club_membership(conv_results$clubs)
#' }

#' Example: Random Forest importance for climate perception
#' @examples
#' \dontrun{
#' # Fit RF model
#' rf_results <- rf_variable_importance(
#'   survey_data,
#'   response = "climate_awareness",
#'   predictors = c("education", "age", "income", "urban")
#' )
#' 
#' # Plot importance
#' ggplot(rf_results$importance, aes(x = reorder(variable, importance), y = importance)) +
#'   geom_col() +
#'   coord_flip() +
#'   theme_publication()
#' }

# =============================================================================
# TOOLKIT INFO
# =============================================================================

#' Print toolkit information
toolkit_info <- function() {
  cat("\n")
  cat("================================================================\n")
  cat("         HTITICH R TOOLKIT v1.0                                 \n")
  cat("================================================================\n")
  cat("\n")
  cat("Author: Mohamed Htitich, PhD\n")
  cat("Email: mohamed.htitich@gmail.com\n")
  cat("\n")
  cat("MODULES:\n")
  cat("  1. Utility Functions       - Data manipulation helpers\n")
  cat("  2. Data Imputation        - Cubic smoothing, interpolation\n")
  cat("  3. Composite Indices      - CESP, JTS, MPI, entropy weights\n")
  cat("  4. Convergence Analysis   - Phillips-Sul club convergence\n")
  cat("  5. Segmented Regression   - Breakpoint detection\n")
  cat("  6. Machine Learning       - Random Forest, Classification Trees\n")
  cat("  7. Multivariate Analysis  - NMDS, ANOSIM\n")
  cat("  8. Eurostat/Spatial       - NUTS data, cascading, GeoJSON\n")
  cat("  9. Visualization          - Choropleths, trajectories, scatter\n")
  cat(" 10. Export Functions       - Tableau, summary tables\n")
  cat("\n")
  cat("To get started: load_packages()\n")
  cat("================================================================\n")
}

# Print info on source
toolkit_info()
