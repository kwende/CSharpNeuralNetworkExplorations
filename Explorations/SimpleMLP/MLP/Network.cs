using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Network
    {
        public List<Layer> Layers { get; set; }

        private Network()
        {

        }

        public static Network BuildNetwork(int inputNeuronCount, int outputNeuronCount, params int[] hiddenLayerCounts)
        {
            Random rand = new Random();

            Layer inputLayer = new Layer();
            inputLayer.LayerIndex = 0;
            inputLayer.Neurons = new List<HiddenNeuron>(inputNeuronCount);
            for (int c = 0; c < inputNeuronCount; c++)
            {
                inputLayer.Neurons[c] = new HiddenNeuron { Bias = rand.NextDouble}
            }
        }
    }
}
