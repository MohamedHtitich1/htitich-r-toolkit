# R Programming Skills Portfolio
## Mohamed Htitich, PhD

---

## Executive Summary

This document summarizes R programming competencies demonstrated through 4+ years of research at the Social Progress Imperative and Palacký University. The accompanying toolkit (`htitich_toolkit.R`) provides production-ready functions covering data science, statistical analysis, machine learning, and spatial visualization.

---

## Core Competencies

### 1. Data Wrangling & Manipulation
**Packages:** dplyr, tidyr, data.table, stringr, purrr

**Demonstrated Skills:**
- Complex data pipelines with chained operations
- Reshaping between wide/long formats
- Multi-source data merging (panel data across years)
- Group-wise operations and rolling calculations
- String manipulation and pattern matching
- Functional programming with map/apply families

**Example Pattern:**
```r
data %>%
  group_by(country) %>%
  mutate(across(vars, ~ zoo::na.approx(.x, na.rm = FALSE))) %>%
  fill(everything(), .direction = "downup") %>%
  mutate(change = value - lag(value, order_by = year))
```

---

### 2. Statistical Modeling
**Packages:** plm, AER, segmented, mgcv, lme4

**Demonstrated Skills:**
- Panel data models (fixed/random effects)
- Segmented regression with breakpoint detection
- Generalized Additive Models (GAM) for imputation
- Instrumental variables estimation
- Robust standard errors (HAC methods)

**Key Applications:**
- SPI-CO2 relationship with structural breaks
- Climate perception determinants across countries
- Time series imputation for incomplete panels

---

### 3. Composite Index Construction
**Packages:** Compind, creditmodel

**Demonstrated Skills:**
- Entropy weighting methods
- Min-max normalization with goal-posts
- Mazziotta-Pareto Index calculation
- Coefficient of variation adjustments
- Multi-dimensional index aggregation

**Indices Developed:**
- Carbon Efficiency of Social Progress (CESP)
- Just Transition Score (JTS)
- Climate Perceptions Index (CPI)
- Material Footprint-adjusted SPI

---

### 4. Convergence Analysis
**Packages:** ConvergenceClubs

**Demonstrated Skills:**
- Phillips-Sul log-t test implementation
- Club convergence identification
- Transition path visualization
- Relative performance metrics
- HAC-robust estimation

**Applications:**
- CESP convergence across 168 countries (1990-2020)
- Regional convergence patterns by income group

---

### 5. Machine Learning
**Packages:** randomForest, partykit, missForest, caret

**Demonstrated Skills:**
- Random Forest for variable importance
- Conditional inference trees (ctree)
- Missing data imputation (missForest)
- Model evaluation metrics
- Cross-validation strategies

**Applications:**
- Climate perception predictor importance (31 countries)
- Classification accuracy >75% for awareness/worry/action
- Survey response imputation

---

### 6. Multivariate Analysis
**Packages:** vegan, indicspecies

**Demonstrated Skills:**
- Non-Metric Multidimensional Scaling (NMDS)
- Bray-Curtis dissimilarity
- ANOSIM group comparisons
- Species/indicator scores
- Ordination visualization

**Applications:**
- Regional patterns in climate perception drivers
- Country clustering by perception profiles

---

### 7. Spatial & GIS Analysis
**Packages:** sf, eurostat, tmap, maps

**Demonstrated Skills:**
- Eurostat API data retrieval
- NUTS hierarchy management (levels 0-3)
- Data cascading (filling gaps from parent regions)
- GeoJSON export for Tableau
- Choropleth mapping (Mollweide projection)

**Applications:**
- EU27 health outcomes mapping (NUTS2)
- Enabling environment indicators
- Interactive Tableau dashboards

---

### 8. Data Visualization
**Packages:** ggplot2, viridis, ggrepel, gghighlight, cowplot

**Demonstrated Skills:**
- Publication-quality figures
- Custom themes and color palettes
- Labeled scatter plots with smart positioning
- Trajectory/time series visualization
- Multi-panel layouts
- Interactive dashboards (Shiny, flexdashboard)

**Style Features:**
- Viridis color schemes for accessibility
- Minimal, clean aesthetics
- Proper annotation and labeling
- Reproducible figure generation

---

### 9. Reproducible Research
**Tools:** RMarkdown, R Projects, Git

**Demonstrated Skills:**
- Parameterized reports
- Modular, reusable functions
- Documentation with roxygen2 style
- Project organization best practices
- Version control workflows

---

### 10. API Integration & Automation
**Platforms:** Eurostat, Facebook Survey (Meta), n8n workflows

**Demonstrated Skills:**
- REST API data retrieval
- Caching strategies
- Error handling for API calls
- Automated data pipelines
- Integration with workflow tools

---

## Technical Metrics

| Metric | Value |
|--------|-------|
| **R Experience** | 4+ years |
| **Lines of Code** | 15,000+ (analysis scripts) |
| **Publications Using R** | 4 peer-reviewed papers |
| **Countries Analyzed** | 170+ |
| **Time Series Span** | 1990-2024 |
| **Survey Responses Processed** | 100,000+ |
| **NUTS2 Regions Mapped** | 240+ |

---

## Package Proficiency Matrix

| Package Category | Packages | Proficiency |
|-----------------|----------|-------------|
| **Data Manipulation** | dplyr, tidyr, data.table | ★★★★★ |
| **Visualization** | ggplot2, viridis, tmap | ★★★★★ |
| **Statistical** | plm, AER, segmented | ★★★★☆ |
| **Machine Learning** | randomForest, partykit | ★★★★☆ |
| **Spatial** | sf, eurostat | ★★★★☆ |
| **Multivariate** | vegan | ★★★★☆ |
| **Index Construction** | Compind, creditmodel | ★★★★★ |
| **Reporting** | RMarkdown, gt | ★★★★☆ |

---

## Sample Output Quality

### Visualizations Created:
- World choropleth maps (Mollweide projection)
- NUTS2 regional health atlases
- Club convergence transition paths
- Classification tree diagrams
- NMDS ordination plots
- Time series trajectories

### Reports Generated:
- SPI technical methodology documents
- JTS country scorecards
- Climate perception index methodology
- EU regional health dashboards

---

## Contact

**Mohamed Htitich, PhD**
- Email: mohamed.htitich@gmail.com
- Location: Prague/Olomouc, Czechia

---

*This portfolio accompanies the `htitich_toolkit.R` comprehensive R library.*
