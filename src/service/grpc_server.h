#pragma once

#include "service/session_manager.h"

#include "analytics.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

/// @brief Thin gRPC adapter that forwards VideoAnalytics RPCs to SessionManager.
///
/// Implements @c videoanalytics::v1::VideoAnalytics without session logic; responses carry
/// application-level outcomes while RPC status remains OK unless transport-layer failures occur.
class GrpcServer final : public videoanalytics::v1::VideoAnalytics::Service {
public:
    /// @param manager Reference to the session manager (must outlive this server).
    explicit GrpcServer(SessionManager& manager);

    /// @brief Starts the gRPC server and blocks until shutdown() is called.
    void run(int port);

    /// @brief Initiates a graceful shutdown (safe to call from a signal handler context).
    void shutdown();

    /// @brief Starts a pipeline session from the incoming request parameters.
    grpc::Status StartSession(grpc::ServerContext* ctx,
                              const videoanalytics::v1::StartSessionRequest* req,
                              videoanalytics::v1::StartSessionResponse* resp) override;

    /// @brief Stops an active session by id.
    grpc::Status StopSession(grpc::ServerContext* ctx,
                             const videoanalytics::v1::StopSessionRequest* req,
                             videoanalytics::v1::StopSessionResponse* resp) override;

    /// @brief Returns snapshots for all tracked sessions.
    grpc::Status ListSessions(grpc::ServerContext* ctx,
                              const videoanalytics::v1::ListSessionsRequest* req,
                              videoanalytics::v1::ListSessionsResponse* resp) override;

    /// @brief Returns status for one session id (including unknown placeholders).
    grpc::Status GetSessionStatus(grpc::ServerContext* ctx,
                                  const videoanalytics::v1::GetSessionStatusRequest* req,
                                  videoanalytics::v1::SessionStatus* resp) override;

private:
    /// @brief Copies session snapshot fields present on the wire into @p dst.
    void fillProtoStatus(const SessionStatus& src, videoanalytics::v1::SessionStatus* dst) const;

    SessionManager& manager_;
    std::unique_ptr<grpc::Server> server_;
};
