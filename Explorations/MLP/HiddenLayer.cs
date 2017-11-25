using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class HiddenLayer : Layer
    {
        private HiddenLayer(int numberOfNeurons, double probabilityOfRandom, Random rand) :
            base(numberOfNeurons, probabilityOfRandom, rand)
        {
        }
        public static HiddenLayer BuildHiddenLayer(Math.RandomNormal rand,
            Layer previousLayer, int numberOfNeurons, double probabilityOfDropout)
        {
            HiddenLayer toReturn = new HiddenLayer(numberOfNeurons, probabilityOfDropout, rand.InternalRandom);

            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(rand, previousLayer));
            }

            return toReturn;
        }
    }
}
