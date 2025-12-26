# Contributing to HTITICH R Toolkit

Thank you for your interest in contributing to this toolkit!

## How to Contribute

### Reporting Issues
- Use the GitHub Issues tab to report bugs or suggest features
- Include a minimal reproducible example when reporting bugs
- Describe expected vs. actual behavior

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-function`)
3. Make your changes
4. Add documentation (roxygen2 style comments)
5. Test your code
6. Commit with clear messages
7. Push and open a Pull Request

### Code Style
- Follow tidyverse style guide
- Use `%>%` pipes for data manipulation
- Include roxygen2-style documentation for all functions
- Add examples in function documentation

### Function Documentation Template
```r
#' Brief title of function
#'
#' Longer description of what the function does.
#'
#' @param param1 Description of parameter 1
#' @param param2 Description of parameter 2
#' @return Description of return value
#' @examples
#' \dontrun{
#' result <- my_function(data, param = "value")
#' }
#' @export
my_function <- function(param1, param2) {
  # Function code
}
```

## Questions?

Feel free to open an issue for any questions about contributing.
