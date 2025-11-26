package dataset

import (
	"math/rand/v2"
)

func VerticalData(samples, classes int) ([][]float64, []uint8) {
	x := make([][]float64, samples*classes)
	y := make([]uint8, samples*classes)
	for i := 0; i < len(x); i++ {
		x[i] = make([]float64, 2)
	}
	for classNumber := 0; classNumber < classes; classNumber++ {
		for i := 0; i < samples; i++ {
			x1 := float64(classNumber) + randomValue()
			x2 := (float64(i)/float64(samples))*4.0 + rand.NormFloat64()*0.2
			x[classNumber*samples+i][0] = x1
			x[classNumber*samples+i][1] = x2
			y[classNumber*samples+i] = uint8(classNumber)
		}
	}
	return x, y
}

func randomValue() float64 {
	return float64(rand.IntN(100)) / 100.0
}
