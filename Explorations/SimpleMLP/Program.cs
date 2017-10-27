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

            Network network = Network.BuildNetwork(1, 1, 5, 15);

            double[,] x = new double[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } };
            double[,] y = new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } };

            network.Train(x, y);

            return;
        }
    }
}
