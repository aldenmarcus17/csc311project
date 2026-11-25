# Run from the repository root after the decision-tree experiment.
library(jsonlite)
directory <- "results/decision_tree"
rows <- fromJSON(file.path(directory, "inner_search_summary.json"), flatten = TRUE)
rows <- subset(rows, eligible & stage == "initial")
depth <- ifelse(is.na(rows$parameters.max_depth), "Unlimited", as.character(rows$parameters.max_depth))
depth <- factor(depth, levels = c("3", "5", "10", "20", "Unlimited"))
accuracy <- tapply(rows$mean_validation_accuracy, list(rows$parameters.min_samples_leaf, depth), mean) * 100
colors <- c("#1a597d", "#a2462a", "#267249", "#6b5085")
dir.create(file.path(directory, "figures"), showWarnings = FALSE)
png(file.path(directory, "figures", "inner_search_accuracy.png"), width = 820, height = 560, res = 110)
par(mar = c(5, 5, 3, 1))
matplot(t(accuracy), type = "b", lty = 1:4, pch = c(16, 17, 15, 18), col = colors,
        axes = FALSE, xlab = "Maximum tree depth", ylab = "Mean inner validation accuracy (%)",
        ylim = c(floor(min(accuracy)) - 1, ceiling(max(accuracy)) + 2),
        main = "Decision tree tuning across respondent folds")
axis(1, seq_len(ncol(accuracy)), colnames(accuracy))
axis(2, las = 1)
box()
legend("bottomright", legend = paste("Minimum leaf size", rownames(accuracy)), col = colors,
       lty = 1:4, pch = c(16, 17, 15, 18), cex = 0.82, bty = "n")
dev.off()
