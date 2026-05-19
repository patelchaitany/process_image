#include "service/grpc_server.h"

#include <cstdio>

GrpcServer::GrpcServer(SessionManager& manager) : manager_(manager) {}

void GrpcServer::run(int port) {
    std::string addr = "0.0.0.0:" + std::to_string(port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
    printf("gRPC server listening on %s\n", addr.c_str());
    server_->Wait();
}

void GrpcServer::shutdown() {
    if (server_) server_->Shutdown();
}

grpc::Status GrpcServer::StartSession(grpc::ServerContext* ctx,
                                      const videoanalytics::v1::StartSessionRequest* req,
                                      videoanalytics::v1::StartSessionResponse* resp) {
    (void)ctx;
    SessionConfig sessionConfig;
    if (req->has_config()) {
        const videoanalytics::v1::SessionConfig& protoConfig = req->config();
        sessionConfig.confidence_threshold = protoConfig.confidence_threshold();
        sessionConfig.match_threshold = protoConfig.match_threshold();
        sessionConfig.max_fps = protoConfig.max_fps();
    }
    StartResult result = manager_.startSession(req->source_uri(), req->callback_url(),
                                               req->session_name(), sessionConfig);
    resp->set_session_id(result.session_id);
    if (result.success) {
        resp->set_status("running");
        resp->clear_error();
    } else {
        resp->set_status("error");
        resp->set_error(result.error);
    }
    return grpc::Status::OK;
}

grpc::Status GrpcServer::StopSession(grpc::ServerContext* ctx,
                                     const videoanalytics::v1::StopSessionRequest* req,
                                     videoanalytics::v1::StopSessionResponse* resp) {
    (void)ctx;
    StopResult result = manager_.stopSession(req->session_id());
    resp->set_success(result.success);
    resp->set_frames_processed(result.frames_processed);
    return grpc::Status::OK;
}

grpc::Status GrpcServer::ListSessions(grpc::ServerContext* ctx,
                                      const videoanalytics::v1::ListSessionsRequest* req,
                                      videoanalytics::v1::ListSessionsResponse* resp) {
    (void)ctx;
    (void)req;
    const std::vector<SessionStatus> sessions = manager_.listSessions();
    for (const SessionStatus& session : sessions) {
        fillProtoStatus(session, resp->add_sessions());
    }
    return grpc::Status::OK;
}

grpc::Status GrpcServer::GetSessionStatus(grpc::ServerContext* ctx,
                                          const videoanalytics::v1::GetSessionStatusRequest* req,
                                          videoanalytics::v1::SessionStatus* resp) {
    (void)ctx;
    const SessionStatus status = manager_.getSessionStatus(req->session_id());
    fillProtoStatus(status, resp);
    return grpc::Status::OK;
}

grpc::Status GrpcServer::RegisterFace(grpc::ServerContext* ctx,
                                      const videoanalytics::v1::RegisterFaceRequest* req,
                                      videoanalytics::v1::RegisterFaceResponse* resp) {
    (void)ctx;
    RegisterFaceResult result = manager_.registerFace(req->name(), req->image_data());
    resp->set_success(result.success);
    resp->set_person_id(result.person_id);
    if (!result.error.empty()) {
        resp->set_error(result.error);
    }
    return grpc::Status::OK;
}

grpc::Status GrpcServer::DetectFaces(grpc::ServerContext* ctx,
                                     const videoanalytics::v1::DetectFacesRequest* req,
                                     videoanalytics::v1::DetectFacesResponse* resp) {
    (void)ctx;
    DetectFacesResult result = manager_.detectFaces(req->image_data(),
                                                    req->confidence_threshold());
    resp->set_success(result.success);
    resp->set_image_width(result.image_width);
    resp->set_image_height(result.image_height);
    if (!result.error.empty()) {
        resp->set_error(result.error);
    }
    for (const auto& det : result.detections) {
        auto* d = resp->add_detections();
        auto* bbox = d->mutable_bbox();
        bbox->set_x1(det.x1);
        bbox->set_y1(det.y1);
        bbox->set_x2(det.x2);
        bbox->set_y2(det.y2);
        d->set_confidence(det.confidence);
    }
    return grpc::Status::OK;
}

void GrpcServer::fillProtoStatus(const SessionStatus& src,
                                 videoanalytics::v1::SessionStatus* dst) const {
    dst->set_session_id(src.session_id);
    dst->set_session_name(src.session_name);
    dst->set_source_uri(src.source_uri);
    dst->set_state(src.state);
    dst->set_frames_processed(src.frames_processed);
    dst->set_frames_dropped(src.frames_dropped);
    dst->set_uptime_seconds(src.uptime_seconds);
}
