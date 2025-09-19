#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>

class StockData {
public:
    double price;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    
    StockData(double p, double v, std::chrono::system_clock::time_point t) 
        : price(p), volume(v), timestamp(t) {}
};

class TechnicalIndicators {
public:
    static std::vector<double> calculateSMA(const std::vector<double>& prices, int period) {
        std::vector<double> sma;
        for (size_t i = period - 1; i < prices.size(); ++i) {
            double sum = 0;
            for (int j = 0; j < period; ++j) {
                sum += prices[i - j];
            }
            sma.push_back(sum / period);
        }
        return sma;
    }
    
    static std::vector<double> calculateRSI(const std::vector<double>& prices, int period = 14) {
        std::vector<double> rsi;
        if (prices.size() < period + 1) return rsi;
        
        std::vector<double> gains, losses;
        for (size_t i = 1; i < prices.size(); ++i) {
            double change = prices[i] - prices[i-1];
            gains.push_back(change > 0 ? change : 0);
            losses.push_back(change < 0 ? -change : 0);
        }
        
        for (size_t i = period - 1; i < gains.size(); ++i) {
            double avgGain = 0, avgLoss = 0;
            for (int j = 0; j < period; ++j) {
                avgGain += gains[i - j];
                avgLoss += losses[i - j];
            }
            avgGain /= period;
            avgLoss /= period;
            
            double rs = (avgLoss == 0) ? 100 : avgGain / avgLoss;
            rsi.push_back(100 - (100 / (1 + rs)));
        }
        return rsi;
    }
};

class MLPredictor {
private:
    std::mt19937 rng;
    double accuracy;
    std::vector<double> weights;
    
public:
    MLPredictor(double acc = 0.68) : accuracy(acc), rng(std::random_device{}()) {
        // Initialize feature weights (simulating learned parameters)
        weights = {0.3, 0.25, 0.2, 0.15, 0.1}; // SMA, RSI, Volume, Momentum, Volatility
    }
    
    double predict(const std::vector<double>& features) {
        // Simulate ML prediction with given accuracy
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Calculate weighted feature score
        double score = 0;
        for (size_t i = 0; i < std::min(features.size(), weights.size()); ++i) {
            score += features[i] * weights[i];
        }
        
        // Add noise and apply accuracy constraint
        double noise = (dist(rng) - 0.5) * 0.2;
        score += noise;
        
        // Simulate accuracy by occasionally flipping prediction
        if (dist(rng) > accuracy) {
            score = -score;
        }
        
        return std::tanh(score); // Return value between -1 and 1
    }
    
    std::vector<double> extractFeatures(const std::vector<StockData>& data, size_t index) {
        if (index < 20 || data.size() < 20) return {};
        
        std::vector<double> prices;
        for (size_t i = 0; i <= index; ++i) {
            prices.push_back(data[i].price);
        }
        
        std::vector<double> features;
        
        // Feature 1: SMA ratio
        auto sma10 = TechnicalIndicators::calculateSMA(prices, 10);
        auto sma20 = TechnicalIndicators::calculateSMA(prices, 20);
        if (!sma10.empty() && !sma20.empty()) {
            features.push_back((sma10.back() / sma20.back()) - 1.0);
        }
        
        // Feature 2: RSI normalized
        auto rsi = TechnicalIndicators::calculateRSI(prices);
        if (!rsi.empty()) {
            features.push_back((rsi.back() - 50.0) / 50.0);
        }
        
        // Feature 3: Volume ratio
        if (index >= 5) {
            double avgVolume = 0;
            for (int i = 0; i < 5; ++i) {
                avgVolume += data[index - i].volume;
            }
            avgVolume /= 5.0;
            features.push_back((data[index].volume / avgVolume) - 1.0);
        }
        
        // Feature 4: Price momentum
        if (index >= 5) {
            double momentum = (prices.back() / prices[prices.size() - 6]) - 1.0;
            features.push_back(momentum);
        }
        
        // Feature 5: Volatility
        if (prices.size() >= 10) {
            double mean = 0;
            for (int i = 0; i < 10; ++i) {
                mean += prices[prices.size() - 1 - i];
            }
            mean /= 10.0;
            
            double variance = 0;
            for (int i = 0; i < 10; ++i) {
                variance += std::pow(prices[prices.size() - 1 - i] - mean, 2);
            }
            variance /= 10.0;
            features.push_back(std::sqrt(variance) / mean);
        }
        
        return features;
    }
};

class Portfolio {
private:
    double cash;
    std::map<std::string, int> positions;
    std::vector<double> portfolioValues;
    std::vector<std::chrono::system_clock::time_point> timestamps;
    
public:
    Portfolio(double initialCash = 100000.0) : cash(initialCash) {}
    
    bool executeTrade(const std::string& symbol, int shares, double price, 
                     std::chrono::system_clock::time_point timestamp) {
        double cost = shares * price;
        
        if (shares > 0) { // Buy
            if (cash >= cost) {
                cash -= cost;
                positions[symbol] += shares;
                return true;
            }
        } else { // Sell
            if (positions[symbol] >= -shares) {
                cash += -cost;
                positions[symbol] += shares;
                return true;
            }
        }
        return false;
    }
    
    void updatePortfolioValue(const std::map<std::string, double>& prices,
                             std::chrono::system_clock::time_point timestamp) {
        double totalValue = cash;
        for (const auto& pos : positions) {
            if (prices.find(pos.first) != prices.end()) {
                totalValue += pos.second * prices.at(pos.first);
            }
        }
        portfolioValues.push_back(totalValue);
        timestamps.push_back(timestamp);
    }
    
    double getCurrentValue(const std::map<std::string, double>& prices) const {
        double totalValue = cash;
        for (const auto& pos : positions) {
            if (prices.find(pos.first) != prices.end()) {
                totalValue += pos.second * prices.at(pos.first);
            }
        }
        return totalValue;
    }
    
    double getInitialValue() const {
        return portfolioValues.empty() ? 0 : portfolioValues[0];
    }
    
    std::vector<double> getPortfolioValues() const { return portfolioValues; }
    std::vector<std::chrono::system_clock::time_point> getTimestamps() const { return timestamps; }
    
    double calculateSharpeRatio() const {
        if (portfolioValues.size() < 2) return 0;
        
        std::vector<double> returns;
        for (size_t i = 1; i < portfolioValues.size(); ++i) {
            double ret = (portfolioValues[i] / portfolioValues[i-1]) - 1.0;
            returns.push_back(ret);
        }
        
        double meanReturn = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0;
        for (double ret : returns) {
            variance += std::pow(ret - meanReturn, 2);
        }
        variance /= returns.size();
        
        double volatility = std::sqrt(variance);
        double riskFreeRate = 0.02 / 252; // Assuming 2% annual risk-free rate, daily
        
        return volatility == 0 ? 0 : (meanReturn - riskFreeRate) / volatility * std::sqrt(252);
    }
};

class MarketDataGenerator {
private:
    std::mt19937 rng;
    
public:
    MarketDataGenerator() : rng(std::random_device{}()) {}
    
    std::vector<StockData> generateData(const std::string& symbol, int days, 
                                       double initialPrice = 100.0) {
        std::vector<StockData> data;
        std::normal_distribution<double> priceDist(0.0, 0.02);
        std::normal_distribution<double> volumeDist(1000000, 200000);
        
        double currentPrice = initialPrice;
        auto currentTime = std::chrono::system_clock::now() - std::chrono::hours(24 * days);
        
        for (int i = 0; i < days; ++i) {
            // Generate multiple ticks per day
            for (int tick = 0; tick < 100; ++tick) {
                double priceChange = priceDist(rng);
                currentPrice *= (1.0 + priceChange);
                currentPrice = std::max(currentPrice, 1.0); // Prevent negative prices
                
                double volume = std::max(100000.0, volumeDist(rng));
                
                data.emplace_back(currentPrice, volume, currentTime);
                currentTime += std::chrono::minutes(4); // ~100 ticks per day
            }
        }
        
        return data;
    }
};

class TradingStrategy {
private:
    MLPredictor predictor;
    double signalThreshold;
    
public:
    TradingStrategy(double accuracy = 0.68, double threshold = 0.3) 
        : predictor(accuracy), signalThreshold(threshold) {}
    
    int generateSignal(const std::vector<StockData>& data, size_t index) {
        auto features = predictor.extractFeatures(data, index);
        if (features.empty()) return 0;
        
        double prediction = predictor.predict(features);
        
        if (prediction > signalThreshold) return 1;  // Buy signal
        if (prediction < -signalThreshold) return -1; // Sell signal
        return 0; // Hold
    }
};

class BacktestEngine {
public:
    struct BacktestResults {
        double totalReturn;
        double sharpeRatio;
        double maxDrawdown;
        int totalTrades;
        int winningTrades;
        std::vector<double> portfolioValues;
        std::vector<std::chrono::system_clock::time_point> timestamps;
    };
    
    static BacktestResults runBacktest(const std::vector<std::string>& symbols,
                                     int days, double accuracy = 0.68) {
        BacktestResults results;
        Portfolio portfolio;
        TradingStrategy strategy(accuracy);
        MarketDataGenerator generator;
        
        // Generate market data for multiple stocks
        std::map<std::string, std::vector<StockData>> marketData;
        for (const auto& symbol : symbols) {
            marketData[symbol] = generator.generateData(symbol, days, 100.0);
        }
        
        // Find the minimum data length
        size_t minLength = marketData.begin()->second.size();
        for (const auto& data : marketData) {
            minLength = std::min(minLength, data.second.size());
        }
        
        int totalTrades = 0;
        int winningTrades = 0;
        std::map<std::string, int> positions;
        std::map<std::string, double> entryPrices;
        
        // Execute trading simulation
        for (size_t i = 50; i < minLength; ++i) { // Start after warm-up period
            std::map<std::string, double> currentPrices;
            
            for (const auto& symbol : symbols) {
                const auto& data = marketData[symbol];
                currentPrices[symbol] = data[i].price;
                
                int signal = strategy.generateSignal(data, i);
                
                if (signal != 0 && positions[symbol] == 0) {
                    // Enter position
                    int shares = static_cast<int>(10000 / data[i].price); // $10k per position
                    if (shares > 0) {
                        bool success = portfolio.executeTrade(symbol, signal * shares, 
                                                            data[i].price, data[i].timestamp);
                        if (success) {
                            positions[symbol] = signal * shares;
                            entryPrices[symbol] = data[i].price;
                            totalTrades++;
                        }
                    }
                }
                else if (signal == 0 && positions[symbol] != 0) {
                    // Exit position
                    bool success = portfolio.executeTrade(symbol, -positions[symbol], 
                                                        data[i].price, data[i].timestamp);
                    if (success) {
                        // Check if winning trade
                        double pnl = positions[symbol] * (data[i].price - entryPrices[symbol]);
                        if (pnl > 0) winningTrades++;
                        positions[symbol] = 0;
                    }
                }
            }
            
            portfolio.updatePortfolioValue(currentPrices, marketData.begin()->second[i].timestamp);
        }
        
        results.totalTrades = totalTrades;
        results.winningTrades = winningTrades;
        results.portfolioValues = portfolio.getPortfolioValues();
        results.timestamps = portfolio.getTimestamps();
        results.sharpeRatio = portfolio.calculateSharpeRatio();
        
        if (!results.portfolioValues.empty()) {
            results.totalReturn = (results.portfolioValues.back() / results.portfolioValues[0]) - 1.0;
            
            // Calculate max drawdown
            double peak = results.portfolioValues[0];
            double maxDD = 0;
            for (double value : results.portfolioValues) {
                if (value > peak) peak = value;
                double drawdown = (peak - value) / peak;
                if (drawdown > maxDD) maxDD = drawdown;
            }
            results.maxDrawdown = maxDD;
        }
        
        return results;
    }
};

class Visualizer {
public:
    static void generateProfitChart(const BacktestEngine::BacktestResults& results, 
                                  const std::string& filename = "profit_chart.html") {
        std::ofstream file(filename);
        
        file << "<!DOCTYPE html>\n<html>\n<head>\n";
        file << "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n";
        file << "<title>Stock AI Trading Results</title>\n</head>\n<body>\n";
        file << "<h1>Stock AI Trading Simulation Results</h1>\n";
        file << "<div id='profitChart' style='width:100%;height:600px;'></div>\n";
        file << "<div id='stats' style='margin-top:20px;'>\n";
        
        file << "<h2>Performance Statistics</h2>\n";
        file << "<p><strong>Total Return:</strong> " << std::fixed << std::setprecision(2) 
             << results.totalReturn * 100 << "%</p>\n";
        file << "<p><strong>Sharpe Ratio:</strong> " << std::fixed << std::setprecision(3) 
             << results.sharpeRatio << "</p>\n";
        file << "<p><strong>Max Drawdown:</strong> " << std::fixed << std::setprecision(2) 
             << results.maxDrawdown * 100 << "%</p>\n";
        file << "<p><strong>Total Trades:</strong> " << results.totalTrades << "</p>\n";
        file << "<p><strong>Win Rate:</strong> " << std::fixed << std::setprecision(1) 
             << (results.totalTrades > 0 ? (double)results.winningTrades / results.totalTrades * 100 : 0) 
             << "%</p>\n";
        file << "</div>\n";
        
        file << "<script>\n";
        file << "var dates = [";
        for (size_t i = 0; i < results.timestamps.size(); ++i) {
            auto time_t = std::chrono::system_clock::to_time_t(results.timestamps[i]);
            file << "'" << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S") << "'";
            if (i < results.timestamps.size() - 1) file << ",";
        }
        file << "];\n";
        
        file << "var values = [";
        for (size_t i = 0; i < results.portfolioValues.size(); ++i) {
            file << results.portfolioValues[i];
            if (i < results.portfolioValues.size() - 1) file << ",";
        }
        file << "];\n";
        
        file << "var profit = values.map(function(val, idx) { return val - values[0]; });\n";
        file << "var trace = {\n";
        file << "  x: dates,\n";
        file << "  y: profit,\n";
        file << "  type: 'scatter',\n";
        file << "  mode: 'lines',\n";
        file << "  name: 'Profit/Loss',\n";
        file << "  line: {color: 'rgb(31, 119, 180)'}\n";
        file << "};\n";
        
        file << "var layout = {\n";
        file << "  title: 'Stock AI Trading Profit Over Time',\n";
        file << "  xaxis: {title: 'Date'},\n";
        file << "  yaxis: {title: 'Profit ($)'},\n";
        file << "  showlegend: true\n";
        file << "};\n";
        
        file << "Plotly.newPlot('profitChart', [trace], layout);\n";
        file << "</script>\n</body>\n</html>";
        
        file.close();
        std::cout << "Profit chart generated: " << filename << std::endl;
    }
};

int main() {
    std::cout << "=== Stock AI Trading Simulation ===\n\n";
    
    int days;
    double accuracy;
    int numStocks;
    
    std::cout << "Enter simulation period (days): ";
    std::cin >> days;
    
    std::cout << "Enter model accuracy (0.0-1.0, default 0.68): ";
    std::cin >> accuracy;
    
    std::cout << "Enter number of stocks to trade (1-20): ";
    std::cin >> numStocks;
    numStocks = std::max(1, std::min(20, numStocks));
    
    // Generate stock symbols
    std::vector<std::string> symbols;
    for (int i = 0; i < numStocks; ++i) {
        symbols.push_back("STOCK" + std::to_string(i + 1));
    }
    
    std::cout << "\nRunning backtest with " << numStocks << " stocks over " 
              << days << " days...\n";
    std::cout << "Model accuracy: " << accuracy * 100 << "%\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run the backtest
    auto results = BacktestEngine::runBacktest(symbols, days, accuracy);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Display results
    std::cout << "=== BACKTEST RESULTS ===\n";
    std::cout << "Processing time: " << duration.count() << " ms\n";
    std::cout << "Data points processed: " << results.portfolioValues.size() << "\n\n";
    
    std::cout << "Performance Metrics:\n";
    std::cout << "- Total Return: " << std::fixed << std::setprecision(2) 
              << results.totalReturn * 100 << "%\n";
    std::cout << "- Sharpe Ratio: " << std::fixed << std::setprecision(3) 
              << results.sharpeRatio << "\n";
    std::cout << "- Max Drawdown: " << std::fixed << std::setprecision(2) 
              << results.maxDrawdown * 100 << "%\n";
    std::cout << "- Total Trades: " << results.totalTrades << "\n";
    std::cout << "- Winning Trades: " << results.winningTrades << "\n";
    std::cout << "- Win Rate: " << std::fixed << std::setprecision(1) 
              << (results.totalTrades > 0 ? (double)results.winningTrades / results.totalTrades * 100 : 0) 
              << "%\n\n";
    
    if (!results.portfolioValues.empty()) {
        double initialValue = results.portfolioValues[0];
        double finalValue = results.portfolioValues.back();
        double totalProfit = finalValue - initialValue;
        
        std::cout << "Portfolio Summary:\n";
        std::cout << "- Initial Value: $" << std::fixed << std::setprecision(2) << initialValue << "\n";
        std::cout << "- Final Value: $" << std::fixed << std::setprecision(2) << finalValue << "\n";
        std::cout << "- Total Profit: $" << std::fixed << std::setprecision(2) << totalProfit << "\n\n";
    }
    
    // Generate profit visualization
    std::cout << "Generating profit visualization...\n";
    Visualizer::generateProfitChart(results);
    
    std::cout << "\nSimulation complete! Open 'profit_chart.html' in your browser to view results.\n";
    
    return 0;
}