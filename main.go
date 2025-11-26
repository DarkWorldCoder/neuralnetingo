package main

import (
	"fmt"

	"github.com/DarkWorldCoder/neuralnetingo/activations"
	"github.com/DarkWorldCoder/neuralnetingo/dataset"
	"github.com/DarkWorldCoder/neuralnetingo/layer"
	"github.com/DarkWorldCoder/neuralnetingo/loss"
	"github.com/DarkWorldCoder/neuralnetingo/utils"
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

}

func main() {

	// RunModel()
	data, _ := dataset.VerticalData(1000, 3)
	dataset.PlotScatterData(data, 3)

}
