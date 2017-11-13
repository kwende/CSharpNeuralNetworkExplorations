using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class Batch
    {
        public Batch()
        {
            Data = new List<TrainingData>();
        }

        public List<TrainingData> Data { get; set; }

        public static Batch[] CreateBatches(List<TrainingData> allData, int numberOfBatches, int batchSize, Random rand)
        {
            // copy the list so as to not muck with the list passed in. 
            List<TrainingData> copyOfData = new List<TrainingData>(allData);
            // randomly sort the list. 
            copyOfData = copyOfData.OrderBy(n => rand.Next()).ToList();
            // return batch size. 
            Batch[] ret = new Batch[numberOfBatches];

            // create numberOfBatches batches. 
            for (int c = 0; c < numberOfBatches; c++)
            {
                Batch newBatch = new Batch();

                // randomly fill the batch. 
                for (int k = 0; k < batchSize; k++)
                {
                    newBatch.Data.Add(allData[rand.Next(0, copyOfData.Count - 1)]);
                }

                ret[c] = newBatch;
            }

            return ret;
        }
    }
}
