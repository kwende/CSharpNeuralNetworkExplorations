using SimpleMLP.MLP;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP
{
    class Program
    {
        static void Main(string[] args)
        {
            // What I cannot create, I do not understand. 
            // ~Richard P. Feynman

            Network network = Network.BuildNetwork(5, 6, 5, 15);

            for (int c = 0; c < 5; c++)
            {
                double[] inputs = new double[5];
                inputs[c] = 1;

                network.SetInputs(inputs);
            }

            return;
        }
    }
}
