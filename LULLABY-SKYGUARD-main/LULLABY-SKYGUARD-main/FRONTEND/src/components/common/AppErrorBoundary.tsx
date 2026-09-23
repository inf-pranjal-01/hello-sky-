import React from 'react';

interface Props { children: React.ReactNode; }
interface State {
  hasError: boolean;
  message: string | null;
}

/** Prevent a transient rendering exception from becoming a blank dashboard. */
export class AppErrorBoundary extends React.Component<Props, State> {
  public state: State = { hasError: false, message: null };

  public static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      // Show the underlying local rendering failure instead of forcing a
      // developer to hunt through a browser console after a blank screen.
      message: error?.message || 'Unknown rendering error',
    };
  }

  public render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <main className="sg-app-error" role="alert">
          <h1>Dashboard needs a refresh</h1>
          <p>A display error occurred; incoming telemetry is not affected.</p>
          {this.state.message && <code className="sg-app-error__detail">{this.state.message}</code>}
          <button type="button" onClick={() => window.location.reload()}>Reload dashboard</button>
        </main>
      );
    }
    return this.props.children;
  }
}
