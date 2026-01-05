form ComputeSlope
    sentence input_file
endform

Read from file: input_file$
# Create pitch object
To Pitch: 0.0, 75, 600

# Select pitch object
selectObject: selected("Pitch")

# Get total duration
tmax = Get end time

# Compute means on first and last 20% of the file
startWindowEnd = tmax * 0.2
endWindowStart = tmax * 0.8

startMean = Get mean: 0, startWindowEnd, "Hertz"
endMean   = Get mean: endWindowStart, tmax, "Hertz"

slope = endMean - startMean

writeInfoLine: slope
