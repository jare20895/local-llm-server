"use client";

import dynamic from "next/dynamic";

const SwaggerUI = dynamic(() => import("swagger-ui-react"), { ssr: false });

export default function SwaggerPage() {
  return (
    <div className="page-wrapper">
      <h1 className="page-title">Vector-Tester API Docs</h1>
      <div className="card" style={{ background: "#fff", color: "#111" }}>
        <SwaggerUI url="/api/docs" />
      </div>
    </div>
  );
}
