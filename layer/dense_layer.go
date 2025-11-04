package layer

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Weights mat.Matrix
	Biases  mat.Dense
	Output  *mat.Dense
}

func (l *Layer) Initialization(n_inputs, n_neurons int) {
	weightsData := make([]float64, n_inputs*n_neurons)
	for i := range weightsData {
		weightsData[i] = rand.NormFloat64() * 0.01
	}
	l.Weights = mat.NewDense(n_neurons, n_inputs, weightsData)

	biasesData := make([]float64, n_neurons)

	l.Biases = *mat.NewDense(1, n_neurons, biasesData)
	l.Biases.Zero()

}

func (l *Layer) Forward(input *mat.Dense) {
	r, _ := input.Dims()
	_, c1 := l.Weights.T().Dims()

	result := mat.NewDense(r, c1, nil)
	result.Product(input, l.Weights.T())
	rows, cols := result.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, result.At(i, j)+l.Biases.At(0, j))
		}
	}
	l.Output = result

}

// utils
func ConvertToMatDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])
	flatData := make([]float64, 0, rows*cols)
	for _, row := range data {
		flatData = append(flatData, row...)
	}
	return mat.NewDense(rows, cols, flatData)
}
