package layer

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Weights mat.Matrix
	Biases  mat.Dense
	Output  mat.Dense
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
	c, _ := l.Weights.Dims()

	result := mat.NewDense(r, c, nil)
	result.Product(input, l.Weights.T())
	rows, cols := result.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, result.At(i, j)+l.Biases.At(0, j))
		}
	}
	l.Output = *result

}

// utils
