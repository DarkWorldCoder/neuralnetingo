package dataset

import (
	"math"
	"math/rand"
)

func SpiralData(points, classes int) ([][]float64, []uint8) {
	X := make([][]float64, points*classes)
	y := make([]uint8, points*classes)
	for classNumber := 0; classNumber < classes; classNumber++ {
		for i := 0; i < points; i++ {
			r := float64(i) / float64(points) * 4.0
			t := float64(classNumber)*4.0 + float64(i)/float64(points)*4.0 + rand.NormFloat64()*0.2
			x1 := r * math.Sin(t)
			x2 := r * math.Cos(t)
			X[classNumber*points+i] = []float64{x1, x2}
			y[classNumber*points+i] = uint8(classNumber)
		}
	}
	return X, y
}
