package main

import (
	"fmt"
	"math"

	"github.com/DarkWorldCoder/neuralnetingo/activations"
	"github.com/DarkWorldCoder/neuralnetingo/dataset"
	"github.com/DarkWorldCoder/neuralnetingo/layer"
	"github.com/DarkWorldCoder/neuralnetingo/loss"
	"github.com/DarkWorldCoder/neuralnetingo/utils"
	"gonum.org/v1/gonum/mat"
)

func RunModel() {
	layer1 := layer.Layer{}
	layer1.Initialization(2, 3)

	X, targetClasses := dataset.SpiralData(100, 3)
	layer1.Forward(utils.ConvertToMatDense(X))
	relu1 := activations.Activation_ReLU{}
	relu1.Forward(&layer1.Output)

	layer2 := layer.Layer{}
	layer2.Initialization(3, 3)
	layer2.Forward(&relu1.Output)
	softmax1 := activations.Activation_Softmax{}
	softmax1.Forward(&layer2.Output)

	lost := loss.CategoricalCrossEntropy{}
	lossValue := lost.Calculate(&softmax1.Output, targetClasses)
	accuracy := loss.CalculateAccuracy(&softmax1.Output, targetClasses)
	fmt.Printf("Loss: %.3f, Accuracy: %.3f\n", lossValue, accuracy)

	lowestLoss := math.MaxFloat64
	bestDense1Weight := mat.DenseCopyOf(layer1.Weights)
	bestDense2Weight := mat.DenseCopyOf(layer2.Weights)
	bestDense1Bias := mat.DenseCopyOf(&layer1.Biases)
	bestDense2Bias := mat.DenseCopyOf(&layer2.Biases)

	for i := 0; i < 10000; i++ {
		layer1.Initialization(2, 3)
		layer1.Forward(utils.ConvertToMatDense(X))
		relu1 := activations.Activation_ReLU{}
		relu1.Forward(&layer1.Output)

		layer2.Initialization(3, 3)
		layer2.Forward(&relu1.Output)
		softmax1 := activations.Activation_Softmax{}
		softmax1.Forward(&layer2.Output)

		lossValue := lost.Calculate(&softmax1.Output, targetClasses)
		accuracy := loss.CalculateAccuracy(&softmax1.Output, targetClasses)

		if lossValue < lowestLoss {
			lowestLoss = lossValue
			bestDense1Weight = mat.DenseCopyOf(layer1.Weights)
			bestDense2Weight = mat.DenseCopyOf(layer2.Weights)
			bestDense1Bias = mat.DenseCopyOf(&layer1.Biases)
			bestDense2Bias = mat.DenseCopyOf(&layer2.Biases)
			fmt.Printf("New best loss: %.3f, Accuracy: %.3f (iteration %d)\n", lossValue, accuracy, i)
		}
	}

	fmt.Println("Training complete.")
	fmt.Printf("Best Loss: %.3f\n", lowestLoss)
	fmt.Println("Best Weights and Biases for Layer 1:")
	fmt.Println(bestDense1Weight)
	fmt.Println(bestDense1Bias)
	fmt.Println("Best Weights and Biases for Layer 2:")
	fmt.Println(bestDense2Weight)
	fmt.Println(bestDense2Bias)
}

func main() {

	RunModel()
	// data, _ := dataset.VerticalData(1000, 3)
	// dataset.PlotScatterData(data, 3)

}
