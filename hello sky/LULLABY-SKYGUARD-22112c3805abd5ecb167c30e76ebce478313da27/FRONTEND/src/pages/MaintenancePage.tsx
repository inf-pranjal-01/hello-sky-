import React from 'react';
import { AlertCircle, RefreshCw, MapPin } from 'lucide-react';
import { useMaintenanceData } from '../hooks/useMaintenanceData';
import { MaintenanceHeader } from '../components/maintenance/MaintenanceHeader';
import { AnomalySelector } from '../components/maintenance/AnomalySelector';
import { TicketPreview } from '../components/maintenance/TicketPreview';
import { TicketSuccessCard } from '../components/maintenance/TicketSuccessCard';
import { ConfirmationDialog } from '../components/common/ConfirmationDialog';
import { Button } from '../components/common/Button';
import { EmptyState } from '../components/common/EmptyState';
import './MaintenancePage.css';

export const MaintenancePage: React.FC = () => {
  const {
    selectedStation,
    anomalies,
    selectedAnomalyId,
    selectedAnomaly,
    isLoadingAnomalies,
    isConfirmModalOpen,
    isSubmitting,
    createdTicket,
    submissionError,
    selectAnomaly,
    openConfirmModal,
    closeConfirmModal,
    confirmCreateTicket,
    resetCreatedTicket,
    refreshAnomalies,
  } = useMaintenanceData();

  return (
    <div className="sg-maintenance-page">
      <MaintenanceHeader
        selectedStation={selectedStation}
        onRefresh={refreshAnomalies}
        isLoading={isLoadingAnomalies}
      />

      {/* No station selected guard */}
      {!selectedStation && !isLoadingAnomalies && (
        <div className="sg-maintenance-page__no-station">
          <EmptyState
            icon={<MapPin size={32} />}
            title="No Station Selected"
            description="Select a weather station from the top navigation to view anomalies and dispatch maintenance tickets."
          />
        </div>
      )}

      {/* Main two-column layout — only shown when a station is selected */}
      {selectedStation && (
        <div className="sg-maintenance-page__body">

          {/* Left column: anomaly selector */}
          <div className="sg-maintenance-page__left">
            <div className="sg-maintenance-page__panel">
              <AnomalySelector
                anomalies={anomalies}
                selectedAnomalyId={selectedAnomalyId}
                onSelect={selectAnomaly}
                isLoading={isLoadingAnomalies}
              />
            </div>
          </div>

          {/* Right column: preview / success / idle state */}
          <div className="sg-maintenance-page__right">
            <div className="sg-maintenance-page__panel">

              {/* Inline submission error */}
              {submissionError && !createdTicket && (
                <div className="sg-maintenance-page__error-banner" role="alert">
                  <AlertCircle size={16} aria-hidden="true" />
                  <div className="sg-maintenance-page__error-body">
                    <span className="sg-maintenance-page__error-msg">{submissionError}</span>
                    {selectedAnomaly && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={openConfirmModal}
                        leftIcon={<RefreshCw size={13} />}
                      >
                        Retry
                      </Button>
                    )}
                  </div>
                </div>
              )}

              {/* Success state */}
              {createdTicket && (
                <TicketSuccessCard
                  ticket={createdTicket}
                  onCreateAnother={resetCreatedTicket}
                />
              )}

              {/* Ticket preview — shown when an anomaly is selected and no ticket yet */}
              {!createdTicket && selectedAnomaly && (
                <TicketPreview
                  anomaly={selectedAnomaly}
                  onCreateTicket={openConfirmModal}
                  isSubmitting={isSubmitting}
                />
              )}

              {/* Idle state: no anomaly selected */}
              {!createdTicket && !selectedAnomaly && !isLoadingAnomalies && anomalies.length > 0 && (
                <EmptyState
                  icon={<AlertCircle size={28} />}
                  title="Select an Anomaly"
                  description="Choose an anomaly from the list on the left to preview and dispatch a maintenance ticket."
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Confirmation dialog */}
      {selectedAnomaly && (
        <ConfirmationDialog
          isOpen={isConfirmModalOpen}
          onClose={closeConfirmModal}
          onConfirm={confirmCreateTicket}
          title="Dispatch Maintenance Ticket"
          message={`Create a maintenance ticket for anomaly ${selectedAnomaly.anomaly_id} at station ${selectedAnomaly.station_id}? This will notify field operations for inspection.`}
          confirmLabel="Dispatch Ticket"
          cancelLabel="Cancel"
          isLoading={isSubmitting}
        />
      )}
    </div>
  );
};
