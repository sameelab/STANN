#"""
#   ___                  _                
#  / _/______ ____  ____(_)__ _______     
# / _/ __/ _ `/ _ \/ __/ (_-</ __/ _ \    
#/_//_/  \_,_/_//_/\__/_/___/\__/\___/    
#  ___ _____(_)__ ___ ____  / /_(_)       
# / _ `/ __/ (_-</ _ `/ _ \/ __/ /        
# \_, /_/ /_/___/\_,_/_//_/\__/_/         
#/___/
#Samee Lab @ Baylor College Of Medicine
#francisco.grisanticanozo@bcm.edu
#Date: 06/2021
#"""

library(plotly)
library(plotrix)
library(spatstat)
library(imager)
library(stringr)
library(sparr)
library(png)
library(cowplot)
library(ggplot2)
library(ks)

obs <- read.csv("./STANN_predictions_sample_slice.csv", stringsAsFactors = F)
obs <- obs[, c("pixel_x", "pixel_y", "STANN_predictions")]

resetPar <- function() {
  dev.new()
  op <- par(no.readonly = TRUE)
  dev.off()
  op
}

for (cy in unique(obs$STANN_predictions)) {
  
  print(cy)
  x <- obs[obs$STANN_predictions == cy, c("pixel_x", "pixel_y")]
  hpi <- Hpi(x = x)
  write.csv(hpi, paste0("./results_ks/", cy, "_hpi.csv"), row.names = F)
  #fhat.pi1 <- kde(x=x, H=hpi, xmin = c(0, 0), xmax = c(2000, 2000), gridsize = rep(200, 2))
  fhat.pi1 <- kde(x=x, H=hpi, xmin = c(0, 0), xmax = c(2000, 2000),gridsize = rep(1000, 2))
  pdf(file = paste0("./results_ks/", cy,"_points.pdf"))
  plot(x$pixel_x, x$pixel_y, axes=T, frame.plot=T, xlab = "spatial1", ylab = "spatial2")
  dev.off()
  
  pdf(file = paste0("./results_ks/", cy,"_ks.pdf"))
  image(fhat.pi1$estimate, useRaster = T, axes = FALSE)
  write.csv(fhat.pi1$estimate, paste0("./results_ks/", cy, "_ks.csv"), row.names = F)
  dev.off()
  
}
par(mfrow = c(1, 1))
