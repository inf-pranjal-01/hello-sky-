import React from 'react';
import { FileText, Printer, Sparkles, RefreshCw } from 'lucide-react';
import { ReportMetadata, Station } from '../../types';
import { Button } from '../common/Button';
import './ReportHeader.css';

export interface ReportHeaderProps {
  station: Station | null;
  metadata: ReportMetadata | null;
  onGenerateReport: () => void;
  onPrintReport: () => void;
  onRefresh: () => void;
  isGenerating?: boolean;
  isLoading?: boolean;
}

export const ReportHeader: React.FC<ReportHeaderProps> = ({
  station,
  metadata,
  onGenerateReport,
  onPrintReport,
  onRefresh,
  isGenerating = false,
  isLoading = false,
}) => {
  const formattedDate = metadata?.generatedAt
    ? new Date(metadata.generatedAt).toLocaleString()
    : new Date().toLocaleString();

  return (
    <header className="sg-report-header">
      <div className="sg-report-header__main">
        <div className="sg-report-header__title-area">
          <div className="sg-report-header__title-row">
            <FileText size={26} className="text-accent" aria-hidden="true" />
            <h1 className="sg-report-header__title">Station Operational Report</h1>
          </div>
          <p className="sg-report-header__subtitle">
            Structured operational review of station telemetry, detected anomalies, sensor hardware
            condition, and regional spatial validation context.
          </p>
        </div>

        {/* Action Controls (Hidden during print) */}
        <div className="sg-report-header__actions no-print">
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            isLoading={isLoading}
            leftIcon={<RefreshCw size={14} />}
            ariaLabel="Refresh raw telemetry data"
          >
            Sync Feeds
          </Button>

          <Button
            variant="secondary"
            size="sm"
            onClick={onGenerateReport}
            isLoading={isGenerating}
            leftIcon={<Sparkles size={14} />}
            ariaLabel="Re-compile operational report"
          >
            Generate Report
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={onPrintReport}
            leftIcon={<Printer size={14} />}
            ariaLabel="Print or Save Report as PDF"
          >
            Print / Save as PDF
          </Button>
        </div>
      </div>

      {/* Report Document Meta Header Banner */}
      {station && metadata && (
        <div className="sg-report-header__meta-banner">
          <div className="sg-report-header__meta-col">
            <span className="sg-report-header__meta-label">Station Identifier</span>
            <span className="sg-report-header__meta-val sg-font-mono">
              {station.name} ({station.station_id})
            </span>
          </div>

          <div className="sg-report-header__meta-col">
            <span className="sg-report-header__meta-label">Observation Window</span>
            <span className="sg-report-header__meta-val">Last {metadata.periodHours} Hours</span>
          </div>

          <div className="sg-report-header__meta-col">
            <span className="sg-report-header__meta-label">Report ID</span>
            <span className="sg-report-header__meta-val sg-font-mono">{metadata.reportId}</span>
          </div>

          <div className="sg-report-header__meta-col">
            <span className="sg-report-header__meta-label">Generated Timestamp</span>
            <span className="sg-report-header__meta-val sg-font-mono">{formattedDate}</span>
          </div>
        </div>
      )}
    </header>
  );
};
