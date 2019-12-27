using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Kinect;
using System.Drawing;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KinectTest
{
    class SavedFrames
    {
        public int Width { get; set; }

        public int Height { get; set; }

        public Dictionary<long, ushort[]> Frames { get; set; }
    }

    class Program
    {
        private static FrameDescription depthFrameDescription;

        private static Dictionary<long, ushort[]> recordedFrames = new Dictionary<long, ushort[]>();

        private static volatile int frameCount;

        private static TimeSpan? firstFrame;
        private static TimeSpan? endTime;
        private static bool shouldRun = true;

        static void Main(string[] args)
        {
            // width: 512
            // height: 424
            // max depth: 4500
            // min depth: 500
            // 16 bit
            KinectSensor camera =  KinectSensor.GetDefault();
            try
            {
                var depthFrameReader = camera.DepthFrameSource.OpenReader();
                depthFrameDescription = camera.DepthFrameSource.FrameDescription;
                depthFrameReader.FrameArrived += DepthFrameReader_FrameArrived;
                camera.Open();
                Console.WriteLine("waiting");
                while (shouldRun) ;
                SavedFrames sf = new SavedFrames();
                sf.Width = depthFrameDescription.Width;
                sf.Height = depthFrameDescription.Height;
                sf.Frames = recordedFrames;
                string json = JsonSerializer.Serialize<SavedFrames>(sf);
                File.WriteAllText("recording.json", json);
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex);
            }
            finally
            {
                camera.Close();
            }
        }
        

        private static void DepthFrameReader_FrameArrived(object sender, DepthFrameArrivedEventArgs e)
        {
            using (DepthFrame depthFrame = e.FrameReference.AcquireFrame())
            {
                if (depthFrame != null)
                {
                    if (!firstFrame.HasValue)
                    {
                        firstFrame = depthFrame.RelativeTime;
                        endTime = firstFrame + TimeSpan.FromSeconds(15);
                    }
                    if (depthFrame.RelativeTime >= endTime)
                    {
                        shouldRun = false;
                    }
                    ushort[] buffer = new ushort[depthFrameDescription.LengthInPixels * depthFrameDescription.BytesPerPixel];
                    depthFrame.CopyFrameDataToArray(buffer);
                    recordedFrames.Add(depthFrame.RelativeTime.Ticks, buffer);
                }
            }
        }
    }
}
