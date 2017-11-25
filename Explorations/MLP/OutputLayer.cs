using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class OutputLayer : Layer
    {
        private OutputLayer(int numberOfNeurons, double probabilityOfDropout, Random rand) :
            base(numberOfNeurons, probabilityOfDropout, rand)
        {
        }
        public static OutputLayer BuildOutputLayer(Math.RandomNormal rand,
            HiddenLayer previousLayer, int numberOfNeurons, double probabilityOfDropout)
        {
            OutputLayer toReturn = new OutputLayer(numberOfNeurons, probabilityOfDropout, rand.InternalRandom);
            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(rand, previousLayer));
            }
            return toReturn;
        }
    }
}
