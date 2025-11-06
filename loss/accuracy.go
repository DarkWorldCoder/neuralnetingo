package loss

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func CalculateAccuracy(prediction *mat.Dense, y []uint8) float64 {
	predictionClasses := make([]uint8, len(y))
	r, _ := prediction.Dims()
	for i := 0; i < r; i++ {
		row := prediction.RawRowView(i)
		maxIndex := floats.MaxIdx(row)
		predictionClasses[i] = uint8(maxIndex)
	}
	correct := 0
	for i := range predictionClasses {
		if predictionClasses[i] == y[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(y))
	return accuracy
}
