# HTITICH R TOOLKIT
## A Comprehensive Library for Sustainable Development Measurement & Analysis

**Author:** Mohamed Htitich, PhD  
**Affiliation:** Social Progress Imperative (2021-2025) | Palacký University Olomouc (2021-2025) | Université Hassan II de Casablanca (2014-2020)
**Version:** 1.0  
**Last Updated:** December 2025

---

## Overview

This toolkit consolidates R functions and workflows developed during 4+ years of research on sustainable development measurement at the Social Progress Imperative. It provides a unified, well-documented codebase for:

- **Composite Index Construction** (JTS, CESP, CPI)
- **Entropy Weighting Methods**
- **Convergence Analysis** (Phillips-Sul club convergence)
- **Panel Data Econometrics**
- **Machine Learning for Survey Data** (Random Forest, Classification Trees)
- **Multivariate Ordination** (NMDS)
- **Spatial Visualization** (Tableau/GeoJSON, Eurostat NUTS mapping)
- **Data Imputation** (cubic smoothing, missForest)

---

## Installation

```r
# Source the toolkit
source("htitich_toolkit.R")

# Load all required packages
load_packages()
```

---

## Module Reference

### 1. Utility Functions

| Function | Description |
|----------|-------------|
| `repeat_before(x)` | Fill NA values forward with last non-NA value |
| `rescale_0_100(x)` | Min-max normalization to 0-100 scale |
| `safe_log(x, base)` | Logarithm that returns NA for non-positive values |
| `geometric_mean(x)` | Calculate geometric mean |
| `keep_eu27(df, extra)` | Filter EU27 countries from NUTS-coded data |

### 2. Data Imputation

| Function | Description |
|----------|-------------|
| `cubic_smoothing_impute()` | GAM-based cubic spline imputation for time series |
| `interp_const_ends()` | Linear interpolation with constant endpoint extension |

**Example:**
```r
# Impute missing GDP values using cubic smoothing
imputed_data <- cubic_smoothing_impute(
  data = gdp_panel,
  var_name = "gdp_percap",
  country_col = "country",
  year_col = "year"
)
```

### 3. Composite Index Construction

| Function | Description |
|----------|-------------|
| `entropy_weights()` | Calculate entropy-based indicator weights |
| `calculate_cesp()` | Carbon Efficiency of Social Progress score |
| `calculate_jts()` | Just Transition Score (combined CO2 + material footprint) |
| `calculate_mpi()` | Mazziotta-Pareto Index |

**Example: Calculate CESP**
```r
results <- calculate_cesp(
  spi_data = spi_scores,
  co2_data = co2_emissions,
  spi_col = "score_spi",
  co2_col = "co2_percap"
)
# Returns: score_cesp (0-100, higher = more efficient)
```

**Example: Calculate JTS**
```r
jts_scores <- calculate_jts(
  data = merged_data,
  spi_col = "score_spi",
  co2_col = "co2_percap",
  matfoot_col = "material_footprint"
)
# Returns: score_cesp, score_mfsp, score_jts
```

### 4. Convergence Analysis

| Function | Description |
|----------|-------------|
| `phillips_sul_convergence()` | Full Phillips-Sul club convergence analysis |
| `extract_club_membership()` | Extract country-club mapping from results |

**Example:**
```r
# Prepare wide-format data (countries × years)
wide_data <- data %>%
  select(country, spiyear, score_cesp) %>%
  pivot_wider(names_from = spiyear, values_from = score_cesp)

# Run convergence analysis
results <- phillips_sul_convergence(
  wide_data,
  unit_col = 1,
  data_cols = 2:32,
  cstar = 0,
  time_trim = 1/3
)

# Get club membership
clubs <- extract_club_membership(results$clubs)
```

### 5. Segmented Regression

| Function | Description |
|----------|-------------|
| `fit_segmented()` | Fit segmented regression with automatic breakpoint detection |
| `extract_segment_params()` | Extract intercepts and slopes for each segment |

**Example:**
```r
# Fit segmented regression for SPI ~ CO2 relationship
seg_model <- fit_segmented(
  data = country_data,
  formula = score_spi ~ co2_percap,
  seg_var = "co2_percap",
  n_breaks = 2,
  type = "bic"
)

# Get segment parameters
params <- extract_segment_params(seg_model)
```

### 6. Machine Learning

| Function | Description |
|----------|-------------|
| `rf_variable_importance()` | Random Forest with importance scores |
| `fit_ctree()` | Conditional inference trees |

**Example: Variable Importance**
```r
rf_results <- rf_variable_importance(
  data = survey_data,
  response = "climate_awareness",
  predictors = c("education", "age", "income", "gender")
)

# Top predictors
head(rf_results$importance)
```

### 7. Multivariate Ordination (NMDS)

| Function | Description |
|----------|-------------|
| `nmds_analysis()` | Full NMDS with site/species scores |
| `test_group_differences()` | ANOSIM test for group differences |

**Example:**
```r
nmds_results <- nmds_analysis(
  data = climate_perception_data,
  species_cols = 5:15,
  group_col = "region",
  distance = "bray"
)

# Check stress value
print(nmds_results$stress)  # Should be < 0.2

# Test regional differences
anosim_test <- test_group_differences(
  data = climate_perception_data,
  species_cols = 5:15,
  group_var = "region"
)
```

### 8. Eurostat/Spatial Functions

| Function | Description |
|----------|-------------|
| `get_nuts_level()` | Fetch Eurostat data at specified NUTS level |
| `build_nuts2_reference()` | Build NUTS2 skeleton with parent codes |
| `cascade_to_nuts2()` | Cascade NUTS0/1 data to NUTS2 level |

**Example: Fetch and Cascade Health Data**
```r
# Fetch NUTS2 hospital beds data
beds_data <- get_nuts_level(
  code = "hlth_rs_bdsrg2",
  level = 2,
  years = 2010:2024
)

# Cascade to fill gaps from NUTS0/1
nuts2_ref <- build_nuts2_reference()
cascaded <- cascade_to_nuts2(
  data = beds_data,
  vars = c("beds", "physicians"),
  nuts2_ref = nuts2_ref
)
```

### 9. Visualization Functions

| Function | Description |
|----------|-------------|
| `world_choropleth()` | World map with country values |
| `theme_publication()` | Publication-ready ggplot theme |
| `trajectory_plot()` | Time series trajectories with labels |
| `scatter_labeled()` | Scatter plot with labeled points |

**Example: Create Choropleth**
```r
world_choropleth(
  data = cesp_2020,
  value_col = "score_cesp",
  code_col = "spicountrycode",
  title = "Carbon Efficiency of Social Progress (2020)",
  palette = "G"
)
```

### 10. Export Functions

| Function | Description |
|----------|-------------|
| `export_for_tableau()` | Export sf data as GeoJSON for Tableau |
| `summary_table()` | Create gt summary statistics table |

---

## Dependencies

The toolkit requires the following R packages:

**Data Manipulation:**
- dplyr, tidyr, data.table, stringr, purrr, readr

**Data Import/Export:**
- readxl, writexl, haven, openxlsx

**Visualization:**
- ggplot2, viridis, ggrepel, gghighlight, cowplot

**Statistical/ML:**
- plm, AER, segmented, randomForest, partykit, missForest

**Multivariate:**
- vegan, indicspecies

**Spatial:**
- sf, tmap, eurostat, maps

**Tables:**
- gt, gtExtras, DT

**Index Construction:**
- Compind, creditmodel, ConvergenceClubs

---

## Research Applications

This toolkit has been used in the following publications:

1. **CESP Paper** (under revision at *Geography and Sustainability*)
   - Functions: `calculate_cesp()`, `phillips_sul_convergence()`, `fit_segmented()`

2. **Climate Perceptions Index** (*Environmental and Sustainability Indicators*)
   - Functions: `rf_variable_importance()`, `nmds_analysis()`, `fit_ctree()`

3. **Just Transition Score** (*Measurement*)
   - Functions: `calculate_jts()`, `entropy_weights()`

4. **Health Atlas** (EU NUTS2 regional analysis)
   - Functions: `get_nuts_level()`, `cascade_to_nuts2()`, `export_for_tableau()`

---

## Complete Workflow Examples

### Example 1: CESP Analysis Pipeline

```r
# 1. Load packages
load_packages()

# 2. Import data
spi_data <- read_xlsx("spi_scores.xlsx")
co2_data <- read_dta("co2_emissions.dta")

# 3. Merge and impute
merged <- spi_data %>%
  left_join(co2_data, by = c("spicountrycode", "spiyear")) %>%
  group_by(spicountrycode) %>%
  mutate(co2_percap = repeat_before(co2_percap))

# 4. Calculate CESP
cesp_results <- calculate_cesp(merged, spi_col = "score_spi", co2_col = "co2_percap")

# 5. Segmented regression for breakpoints
seg_model <- fit_segmented(
  cesp_results, 
  formula = score_spi ~ co2_percap,
  seg_var = "co2_percap"
)

# 6. Convergence analysis
wide_cesp <- cesp_results %>%
  select(country, spicountrycode, spiyear, score_cesp) %>%
  pivot_wider(names_from = spiyear, values_from = score_cesp)

conv_results <- phillips_sul_convergence(wide_cesp, unit_col = 1, data_cols = 3:33)
clubs <- extract_club_membership(conv_results$clubs)

# 7. Visualize
world_choropleth(
  cesp_results %>% filter(spiyear == 2020),
  value_col = "score_cesp",
  title = "CESP 2020"
)
```

### Example 2: Climate Perception Analysis

```r
# 1. Random Forest for variable importance
rf_awareness <- rf_variable_importance(
  survey_data,
  response = "climate_awareness",
  predictors = c("education", "age", "gender", "income", "urban", "gov_trust")
)

# 2. NMDS ordination
nmds_results <- nmds_analysis(
  data = country_importance_data,
  species_cols = 5:15,
  group_col = "region"
)

# 3. Test regional differences
anosim_result <- test_group_differences(
  country_importance_data,
  species_cols = 5:15,
  group_var = "region"
)

# 4. Classification tree for subgroup analysis
tree_model <- fit_ctree(
  survey_data,
  response = "climate_action",
  predictors = c("awareness", "worry", "gov_priority", "beliefs"),
  max_depth = 3
)
```

### Example 3: Eurostat NUTS2 Health Mapping

```r
# 1. Fetch data from Eurostat API
beds <- get_nuts_level("hlth_rs_bdsrg2", level = 2, years = 2010:2024)
physicians <- get_nuts_level("hlth_rs_physreg", level = 2, years = 2010:2024)

# 2. Build reference skeleton
nuts2_ref <- build_nuts2_reference()

# 3. Process and cascade
processed <- beds %>%
  filter(unit == "P_HTHAB") %>%
  select(geo, year = time, beds = values) %>%
  left_join(physicians %>% filter(unit == "P_HTHAB") %>% 
              select(geo, year = time, physicians = values))

cascaded <- cascade_to_nuts2(processed, vars = c("beds", "physicians"), nuts2_ref = nuts2_ref)

# 4. Score and export
final_data <- cascaded %>%
  keep_eu27() %>%
  mutate(
    beds_score = rescale_0_100(beds),
    physicians_score = rescale_0_100(physicians),
    ee_composite = (beds_score + physicians_score) / 2
  )

# 5. Export for Tableau
export_for_tableau(final_data, "eu_health_nuts2.geojson")
```

---

## Skills Demonstrated

| Category | Techniques |
|----------|------------|
| **Data Wrangling** | dplyr pipelines, tidyr reshaping, data.table, joins/merges |
| **Statistical Modeling** | OLS, panel models (plm), segmented regression, GAM |
| **Machine Learning** | Random Forest, Classification Trees, missForest imputation |
| **Index Construction** | Entropy weighting, min-max normalization, Mazziotta-Pareto |
| **Convergence Analysis** | Phillips-Sul log-t test, club formation algorithms |
| **Multivariate Statistics** | NMDS ordination, ANOSIM, Bray-Curtis dissimilarity |
| **Spatial Analysis** | sf objects, GeoJSON export, NUTS hierarchies, choropleths |
| **API Integration** | Eurostat API (eurostat package) |
| **Visualization** | ggplot2, viridis palettes, ggrepel labeling, tmap |
| **Reproducibility** | RMarkdown, parameterized reports, modular functions |

---

## License

MIT License - Free to use, modify, and distribute with attribution.

---

## Contact

**Mohamed Htitich, PhD**  
- Email: m.ahtitich@gmail.com or (@outlook.com) 
- ORCID: https://orcid.org/0000-0002-8732-6286 
- GitHub: You're here! :)
- Location: Prague/Olomouc/Casablanca, Czechia and Morocco

---

*"Measuring not only what matters, but what matters most for sustainable development"*
