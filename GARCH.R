# ----------------------------- #
# VOLATILITY FORECASTING ON IRREGULAR TIME SERIES
# Using Lomb-Scargle Periodogram + GARCH(1,1) in R
# Author: Vedang Gotmare
# ----------------------------- #

# 1. Load required libraries
library(quantmod)      # for financial data
library(lomb)          # Lomb–Scargle periodogram
library(rugarch)       # GARCH modeling
library(tseries)       # additional time series tools
library(xts)

# 2. Get stock data from Yahoo Finance
getSymbols("TSLA", src = "yahoo", from = "2020-01-01", to = Sys.Date())
tsla <- na.omit(Cl(TSLA))  # use closing price

# 3. Simulate irregular sampling (drop 40% of dates randomly)
set.seed(42)
irregular_idx <- sort(sample(1:length(tsla), size = floor(0.6 * length(tsla)), replace = FALSE))
tsla_irregular <- tsla[irregular_idx]

# 4. Convert index to numeric time for Lomb–Scargle
price <- as.numeric(tsla_irregular)
time <- as.numeric(index(tsla_irregular))  # numeric days since 1970

# 5. Apply Lomb–Scargle Periodogram
cat("Running Lomb–Scargle Periodogram...\n")
ls_result <- lomb::lsp(price, times = time, from = 0.01, to = 1, type = "frequency", plot = TRUE)
dominant_freq <- ls_result$peak.at[1]
cat("Dominant frequency detected:", dominant_freq, "\n")

# 6. Calculate log returns for GARCH
log_returns <- diff(log(tsla_irregular))[-1]

# 7. Define GARCH(1,1) model spec
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# 8. Fit GARCH model
cat("Fitting GARCH(1,1) Model...\n")
garch_fit <- ugarchfit(spec = garch_spec, data = log_returns)

# 9. Plot diagnostics
plot(garch_fit)

# 10. Summary output
cat("\nModel Fit Summary:\n")
show(garch_fit)
