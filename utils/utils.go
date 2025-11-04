package utils

import "gonum.org/v1/gonum/mat"

func ConvertToMatDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])
	flatData := make([]float64, 0, rows*cols)
	for _, row := range data {
		flatData = append(flatData, row...)
	}
	return mat.NewDense(rows, cols, flatData)
}
