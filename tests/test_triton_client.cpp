#include <gtest/gtest.h>
#include "inference/triton_client.h"

TEST(TritonClientTest, InferWithoutConnection) {
    TritonClient client;
    auto result = client.infer("model", "input_shm", {1, 3, 640, 640},
                                "output_shm", {1, 8400, 5});
    EXPECT_FALSE(result.isSuccess);
    EXPECT_FALSE(result.errorMsg.empty());
}

TEST(TritonClientTest, NotConnectedByDefault) {
    TritonClient client;
    EXPECT_FALSE(client.isConnected());
}

TEST(TritonClientTest, RegisterShmWithoutConnection) {
    TritonClient client;
    EXPECT_FALSE(client.registerCudaShm("test", nullptr, 1024));
}

TEST(TritonClientTest, InferDirectWithoutConnection) {
    TritonClient client;
    float dummy[1] = {0.0f};
    auto result = client.inferDirect("model", dummy, {1, 3, 640, 640}, {1, 8400, 5});
    EXPECT_FALSE(result.isSuccess);
    EXPECT_FALSE(result.errorMsg.empty());
}

// Integration tests requiring running Triton server
TEST(TritonClientTest, DISABLED_ConnectToLiveServer) {
    TritonClient client;
    ASSERT_TRUE(client.connect("localhost:8001"));
    EXPECT_TRUE(client.isConnected());
    client.disconnect();
    EXPECT_FALSE(client.isConnected());
}

TEST(TritonClientTest, DISABLED_InferOnLiveServer) {
    TritonClient client;
    ASSERT_TRUE(client.connect("localhost:8001"));

    auto result = client.infer("yolo26_face", "yolo_input_shm", {1, 3, 640, 640},
                                "yolo_output_shm", {1, 8400, 5});
    EXPECT_TRUE(result.isSuccess);
    EXPECT_EQ(result.outputShape.size(), 3);

    client.disconnect();
}
