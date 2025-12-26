#' ============================================================================
#' HTITICH R TOOLKIT - EXAMPLE WORKFLOWS
#' Practical demonstrations of key functions
#' ============================================================================

# Source the main toolkit
source("htitich_toolkit.R")

# =============================================================================
# EXAMPLE 1: COMPOSITE INDEX CONSTRUCTION (CESP/JTS)
# =============================================================================

#' This workflow demonstrates:
#' - Data merging and cleaning
#' - CESP calculation
#' - JTS calculation
#' - Visualization

example_cesp_jts <- function() {
  cat("\n=== EXAMPLE 1: CESP & JTS Calculation ===\n")
  
  # Simulate data (replace with actual data import)
  set.seed(42)
  n_countries <- 50
  n_years <- 10
  
  data <- expand.grid(
    country = paste0("Country_", 1:n_countries),
    year = 2010:2019
  ) %>%
    mutate(
      spicountrycode = paste0("C", sprintf("%02d", as.numeric(factor(country)))),
      score_spi = rnorm(n(), 60, 15) %>% pmax(20) %>% pmin(95),
      co2_percap = rexp(n(), 0.2) + 1,
      matfoot_percap = rexp(n(), 0.15) + 3,
      gdp_percap = exp(rnorm(n(), 9, 1.2))
    )
  
  # Calculate CESP
  cesp_results <- calculate_cesp(
    data,
    spi_col = "score_spi",
    co2_col = "co2_percap",
    country_col = "country",
    year_col = "year"
  )
  
  cat("CESP Statistics (2019):\n")
  print(summary(cesp_results$score_cesp[cesp_results$year == 2019]))
  
  # Calculate JTS
  jts_results <- calculate_jts(
    data,
    spi_col = "score_spi",
    co2_col = "co2_percap",
    matfoot_col = "matfoot_percap"
  )
  
  cat("\nJTS Statistics (2019):\n")
  print(summary(jts_results$score_jts[jts_results$year == 2019]))
  
  # Return results
  invisible(list(cesp = cesp_results, jts = jts_results))
}

# =============================================================================
# EXAMPLE 2: CONVERGENCE ANALYSIS
# =============================================================================

#' This workflow demonstrates:
#' - Data preparation for convergence analysis
#' - Phillips-Sul club convergence
#' - Transition path visualization

example_convergence <- function() {
  cat("\n=== EXAMPLE 2: Club Convergence Analysis ===\n")
  
  # Simulate CESP panel data
  set.seed(123)
  n_countries <- 30
  years <- 1990:2020
  
  # Create convergence clubs pattern
  data <- data.frame(
    country = rep(paste0("Country_", 1:n_countries), each = length(years)),
    year = rep(years, n_countries)
  ) %>%
    group_by(country) %>%
    mutate(
      # Assign to clubs
      club = case_when(
        as.numeric(factor(country)) <= 10 ~ "Club1",
        as.numeric(factor(country)) <= 20 ~ "Club2",
        TRUE ~ "Club3"
      ),
      # Generate converging paths within clubs
      base = case_when(
        club == "Club1" ~ 75,
        club == "Club2" ~ 55,
        club == "Club3" ~ 35
      ),
      score_cesp = base + 
        (year - 1990) * rnorm(1, 0.3, 0.1) + 
        rnorm(n(), 0, 3)
    ) %>%
    ungroup()
  
  # Prepare wide format
  wide_data <- data %>%
    select(country, year, score_cesp) %>%
    pivot_wider(names_from = year, values_from = score_cesp)
  
  cat("Data prepared for convergence analysis:\n")
  cat("- Countries:", n_countries, "\n")
  cat("- Years:", length(years), "\n")
  
  # Note: Full convergence analysis requires ConvergenceClubs package
  # This demonstrates the data preparation workflow
  
  # Visualize transition paths
  p <- ggplot(data, aes(x = year, y = score_cesp, group = country, color = club)) +
    geom_line(alpha = 0.5) +
    geom_smooth(aes(group = club), method = "loess", se = FALSE, size = 1.5) +
    scale_color_viridis_d() +
    labs(
      title = "Simulated Convergence Paths",
      x = "Year",
      y = "CESP Score",
      color = "Club"
    ) +
    theme_publication()
  
  print(p)
  
  invisible(list(data = data, wide = wide_data))
}

# =============================================================================
# EXAMPLE 3: SEGMENTED REGRESSION
# =============================================================================

#' This workflow demonstrates:
#' - Fitting segmented regression
#' - Breakpoint identification
#' - Segment visualization

example_segmented <- function() {
  cat("\n=== EXAMPLE 3: Segmented Regression ===\n")
  
  # Generate data with two breakpoints
  set.seed(456)
  n <- 200
  
  data <- data.frame(
    co2_percap = c(
      runif(n/3, 0.5, 2),      # Low emissions
      runif(n/3, 2, 10),       # Medium emissions
      runif(n/3, 10, 30)       # High emissions
    )
  ) %>%
    mutate(
      score_spi = case_when(
        co2_percap <= 2 ~ 30 + 15 * co2_percap + rnorm(n(), 0, 3),
        co2_percap <= 10 ~ 60 + 2 * (co2_percap - 2) + rnorm(n(), 0, 3),
        TRUE ~ 76 + 0.2 * (co2_percap - 10) + rnorm(n(), 0, 3)
      )
    )
  
  # Fit initial OLS
  lm_model <- lm(score_spi ~ co2_percap, data = data)
  
  cat("OLS R-squared:", round(summary(lm_model)$r.squared, 3), "\n")
  
  # Fit segmented regression
  require(segmented)
  seg_model <- segmented(
    lm_model,
    seg.Z = ~ co2_percap,
    psi = list(co2_percap = c(2, 10))
  )
  
  cat("\nBreakpoints identified:\n")
  print(seg_model$psi)
  
  # Visualize
  data$fitted_seg <- fitted(seg_model)
  
  p <- ggplot(data, aes(x = co2_percap, y = score_spi)) +
    geom_point(alpha = 0.5, color = "grey50") +
    geom_line(aes(y = fitted_seg), color = "firebrick", size = 1.2) +
    geom_vline(xintercept = seg_model$psi[, "Est."], linetype = "dashed", color = "blue") +
    labs(
      title = "SPI vs CO2 Emissions: Segmented Relationship",
      subtitle = "Breakpoints identified at ~2 and ~10 tonnes per capita",
      x = "CO2 per capita (tonnes)",
      y = "SPI Score"
    ) +
    theme_publication()
  
  print(p)
  
  invisible(seg_model)
}

# =============================================================================
# EXAMPLE 4: RANDOM FOREST VARIABLE IMPORTANCE
# =============================================================================

#' This workflow demonstrates:
#' - Random Forest for classification
#' - Variable importance extraction
#' - Importance visualization

example_random_forest <- function() {
  cat("\n=== EXAMPLE 4: Random Forest Variable Importance ===\n")
  
  # Simulate climate perception survey data
  set.seed(789)
  n <- 1000
  
  survey_data <- data.frame(
    # Response
    climate_awareness = factor(sample(c("Aware", "Unaware"), n, 
                                      prob = c(0.7, 0.3), replace = TRUE)),
    # Predictors
    education = factor(sample(c("Primary", "Secondary", "Tertiary"), n, 
                              prob = c(0.2, 0.4, 0.4), replace = TRUE)),
    age_group = factor(sample(c("18-29", "30-49", "50+"), n, replace = TRUE)),
    income = rnorm(n, 50000, 20000) %>% pmax(10000),
    urban = factor(sample(c("Urban", "Rural"), n, prob = c(0.6, 0.4), replace = TRUE)),
    gov_trust = sample(1:10, n, replace = TRUE),
    media_exposure = sample(1:5, n, replace = TRUE)
  )
  
  # Make awareness more likely for educated, younger, urban
  survey_data <- survey_data %>%
    mutate(
      aware_prob = 0.5 + 
        0.15 * (education == "Tertiary") +
        0.1 * (age_group == "18-29") +
        0.05 * (urban == "Urban") +
        0.02 * media_exposure,
      climate_awareness = factor(ifelse(runif(n) < aware_prob, "Aware", "Unaware"))
    ) %>%
    select(-aware_prob)
  
  # Fit Random Forest
  require(randomForest)
  
  rf_model <- randomForest(
    climate_awareness ~ education + age_group + income + urban + gov_trust + media_exposure,
    data = survey_data,
    ntree = 500,
    importance = TRUE
  )
  
  # Extract importance
  importance_df <- data.frame(
    variable = rownames(importance(rf_model)),
    MeanDecreaseGini = importance(rf_model)[, "MeanDecreaseGini"]
  ) %>%
    arrange(desc(MeanDecreaseGini)) %>%
    mutate(importance_scaled = rescale_0_100(MeanDecreaseGini))
  
  cat("Variable Importance (Top to Bottom):\n")
  print(importance_df)
  
  # Visualize
  p <- ggplot(importance_df, aes(x = reorder(variable, MeanDecreaseGini), 
                                  y = MeanDecreaseGini)) +
    geom_col(fill = "dodgerblue4") +
    coord_flip() +
    labs(
      title = "Climate Awareness: Variable Importance",
      x = "",
      y = "Mean Decrease in Gini"
    ) +
    theme_publication()
  
  print(p)
  
  invisible(list(model = rf_model, importance = importance_df))
}

# =============================================================================
# EXAMPLE 5: NMDS ORDINATION
# =============================================================================

#' This workflow demonstrates:
#' - NMDS analysis for multivariate data
#' - Group comparisons with ANOSIM
#' - Ordination plot

example_nmds <- function() {
  cat("\n=== EXAMPLE 5: NMDS Ordination ===\n")
  
  # Simulate country-level climate perception importance scores
  set.seed(101)
  n_countries <- 30
  
  # Create regional patterns
  data <- data.frame(
    country = paste0("Country_", 1:n_countries),
    region = rep(c("Europe", "Asia", "Americas"), each = 10),
    # Variable importance scores (0-100)
    Education = rnorm(n_countries, 60, 15),
    Age = rnorm(n_countries, 50, 20),
    Income = rnorm(n_countries, 55, 18),
    Urban = rnorm(n_countries, 45, 15),
    GovTrust = rnorm(n_countries, 40, 20),
    MediaExposure = rnorm(n_countries, 65, 12)
  ) %>%
    mutate(across(Education:MediaExposure, ~ pmax(0, pmin(100, .x))))
  
  # Add regional differences
  data <- data %>%
    mutate(
      Education = Education + ifelse(region == "Europe", 10, 
                                     ifelse(region == "Americas", 5, -5)),
      GovTrust = GovTrust + ifelse(region == "Asia", 15, -5)
    )
  
  # Run NMDS
  require(vegan)
  
  dist_matrix <- vegdist(data[, 3:8], method = "bray")
  nmds <- metaMDS(dist_matrix, k = 2, trymax = 100)
  
  cat("NMDS Stress:", round(nmds$stress, 3), "\n")
  cat("(Stress < 0.2 indicates good fit)\n")
  
  # Extract scores
  site_scores <- as.data.frame(scores(nmds, display = "sites"))
  site_scores$country <- data$country
  site_scores$region <- data$region
  
  # ANOSIM test
  anosim_result <- anosim(dist_matrix, data$region, permutations = 999)
  
  cat("\nANOSIM R statistic:", round(anosim_result$statistic, 3), "\n")
  cat("ANOSIM p-value:", anosim_result$signif, "\n")
  
  # Visualize
  centroids <- site_scores %>%
    group_by(region) %>%
    summarise(NMDS1 = mean(NMDS1), NMDS2 = mean(NMDS2))
  
  p <- ggplot(site_scores, aes(x = NMDS1, y = NMDS2, color = region)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_point(data = centroids, size = 5, shape = 18) +
    geom_text(aes(label = country), size = 2.5, vjust = -0.8) +
    scale_color_viridis_d() +
    labs(
      title = "NMDS: Climate Perception Drivers by Region",
      subtitle = sprintf("Stress = %.3f | ANOSIM R = %.3f", nmds$stress, anosim_result$statistic),
      color = "Region"
    ) +
    theme_publication()
  
  print(p)
  
  invisible(list(nmds = nmds, site_scores = site_scores, anosim = anosim_result))
}

# =============================================================================
# EXAMPLE 6: EUROSTAT DATA PIPELINE
# =============================================================================

#' This workflow demonstrates:
#' - Fetching Eurostat data
#' - NUTS level cascading
#' - Scoring and visualization

example_eurostat <- function() {
  cat("\n=== EXAMPLE 6: Eurostat Data Pipeline (Simulated) ===\n")
  
  # Simulate NUTS2 health data (actual implementation would use eurostat package)
  set.seed(202)
  
  # EU countries
  eu_countries <- c("AT", "BE", "BG", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", 
                    "GR", "HR", "HU", "IE", "IT", "LT", "LV", "NL", "PL", "PT", 
                    "RO", "SE", "SI", "SK")
  
  # Generate NUTS2 codes
  nuts2_codes <- unlist(lapply(eu_countries, function(x) paste0(x, sprintf("%02d", 1:5))))
  
  data <- expand.grid(
    geo = nuts2_codes,
    year = 2015:2023
  ) %>%
    mutate(
      country = substr(geo, 1, 2),
      beds_per_100k = rnorm(n(), 450, 150) %>% pmax(100),
      physicians_per_100k = rnorm(n(), 350, 100) %>% pmax(50),
      mortality_rate = rnorm(n(), 1000, 200) %>% pmax(400)
    )
  
  # Score indicators
  data_scored <- data %>%
    filter(year == 2023) %>%
    mutate(
      beds_score = rescale_0_100(beds_per_100k),
      physicians_score = rescale_0_100(physicians_per_100k),
      mortality_score = rescale_0_100(-mortality_rate),  # Lower is better
      composite_score = (beds_score + physicians_score + mortality_score) / 3
    )
  
  cat("Scored", nrow(data_scored), "NUTS2 regions\n")
  cat("\nComposite Score Statistics:\n")
  print(summary(data_scored$composite_score))
  
  # Country-level summary
  country_summary <- data_scored %>%
    group_by(country) %>%
    summarise(
      mean_score = mean(composite_score),
      min_score = min(composite_score),
      max_score = max(composite_score),
      range = max_score - min_score
    ) %>%
    arrange(desc(mean_score))
  
  cat("\nTop 5 Countries by Mean Composite Score:\n")
  print(head(country_summary, 5))
  
  invisible(list(data = data_scored, summary = country_summary))
}

# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================

run_all_examples <- function() {
  cat("\n")
  cat("================================================================\n")
  cat("    RUNNING ALL HTITICH R TOOLKIT EXAMPLES\n")
  cat("================================================================\n")
  
  # Example 1
  example_cesp_jts()
  readline(prompt = "Press [Enter] to continue to Example 2...")
  
  # Example 2
  example_convergence()
  readline(prompt = "Press [Enter] to continue to Example 3...")
  
  # Example 3
  example_segmented()
  readline(prompt = "Press [Enter] to continue to Example 4...")
  
  # Example 4
  example_random_forest()
  readline(prompt = "Press [Enter] to continue to Example 5...")
  
  # Example 5
  example_nmds()
  readline(prompt = "Press [Enter] to continue to Example 6...")
  
  # Example 6
  example_eurostat()
  
  cat("\n")
  cat("================================================================\n")
  cat("    ALL EXAMPLES COMPLETED\n")
  cat("================================================================\n")
}

# Uncomment to run all examples:
# run_all_examples()
