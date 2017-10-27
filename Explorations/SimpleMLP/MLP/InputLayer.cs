using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class InputLayer : Layer
    {
        private InputLayer() : base()
        {
        }
        public static InputLayer BuildInputLayer(Math.RandomNormal rand, int number)
        {
            InputLayer toReturn = new InputLayer();
            for (int c = 0; c < number; c++)
            {
                toReturn.Neurons.Add(new InputNeuron());
            }
            return toReturn;
        }
    }
}
