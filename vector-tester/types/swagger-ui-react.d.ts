declare module "swagger-ui-react" {
  import * as React from "react";

  export interface SwaggerUIProps {
    url?: string;
    spec?: Record<string, any>;
    docExpansion?: "list" | "full" | "none";
    defaultModelsExpandDepth?: number;
    defaultModelExpandDepth?: number;
  }

  export default class SwaggerUI extends React.Component<SwaggerUIProps> {}
}
