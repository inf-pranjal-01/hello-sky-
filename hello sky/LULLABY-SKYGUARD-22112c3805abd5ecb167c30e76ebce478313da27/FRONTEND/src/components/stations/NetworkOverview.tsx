import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CircleMarker, MapContainer, TileLayer, Tooltip as LeafletTooltip, useMap, useMapEvents } from 'react-leaflet';
import type { LatLngBoundsExpression, Map as LeafletMap } from 'leaflet';
import { Network, ZoomIn, ZoomOut, Maximize2, MapPinned } from 'lucide-react';
import { Station } from '../../types';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { Button } from '../common/Button';
import 'leaflet/dist/leaflet.css';
import './NetworkOverview.css';

export interface NetworkOverviewProps { stations: Station[]; selectedStation: Station | null; onSelectStation: (station: Station) => void; isLoading?: boolean; className?: string; }
const CITY_FROM_CODE: Record<string, string> = { DEL: 'Delhi', MUM: 'Mumbai', CHN: 'Chennai', KOL: 'Kolkata', BHO: 'Bhopal' };
const statusColor = (status: string) => status === 'NORMAL' ? '#10b981' : status === 'WARNING' ? '#f59e0b' : status === 'CRITICAL' ? '#ef4444' : '#64748b';
const stationCity = (station: Station) => CITY_FROM_CODE[station.station_id.match(/AWS-([A-Z]{3})/i)?.[1].toUpperCase() ?? ''] ?? station.name;

function StationViewport({ stations, selectedStation }: { stations: Station[]; selectedStation: Station | null }) {
  const map = useMap();
  const stationKey = stations.map(({ station_id, lat, lon }) => `${station_id}:${lat}:${lon}`).join('|');
  useEffect(() => {
    if (!stations.length) return;
    const bounds: LatLngBoundsExpression = stations.map(({ lat, lon }) => [lat, lon]);
    map.fitBounds(bounds, { padding: [48, 48], maxZoom: 6, animate: false });
  }, [map, stationKey]);
  useEffect(() => {
    if (selectedStation && Number.isFinite(selectedStation.lat) && Number.isFinite(selectedStation.lon)) {
      map.flyTo([selectedStation.lat, selectedStation.lon], Math.max(map.getZoom(), 8), { duration: 0.45 });
    }
  }, [map, selectedStation?.station_id]);
  return null;
}

function ZoomReporter({ onZoomChange }: { onZoomChange: (zoom: number) => void }) {
  const map = useMapEvents({ zoomend: () => onZoomChange(map.getZoom()) });
  useEffect(() => onZoomChange(map.getZoom()), [map, onZoomChange]);
  return null;
}

export const NetworkOverview: React.FC<NetworkOverviewProps> = ({ stations, selectedStation, onSelectStation, isLoading = false, className = '' }) => {
  const mapRef = useRef<LeafletMap | null>(null);
  const [zoom, setZoom] = useState(5);
  const validStations = useMemo(() => stations.filter((station) => Number.isFinite(station.lat) && Number.isFinite(station.lon)), [stations]);
  const fitAll = () => {
    if (validStations.length) mapRef.current?.fitBounds(validStations.map(({ lat, lon }) => [lat, lon]), { padding: [48, 48], maxZoom: 6 });
  };
  if (isLoading) return <Card variant="glass" className={`sg-network-overview-card ${className}`}><Skeleton width="100%" height="360px" /></Card>;

  return <Card variant="glass" className={`sg-network-overview-card ${className}`} role="region" aria-label="Interactive station network map">
    <div className="sg-network-overview__header">
      <div className="sg-network-overview__title-group"><Network size={18} className="text-accent" aria-hidden="true" /><h3 className="sg-network-overview__title">Observatory Network Map</h3></div>
      <div className="sg-network-overview__zoom-controls" aria-label="Map controls">
        <Button variant="outline" size="sm" onClick={() => mapRef.current?.zoomOut()} ariaLabel="Zoom out" leftIcon={<ZoomOut size={14} />}>Zoom out</Button>
        <span className="sg-network-overview__zoom-level">Zoom {zoom}</span>
        <Button variant="outline" size="sm" onClick={() => mapRef.current?.zoomIn()} ariaLabel="Zoom in" leftIcon={<ZoomIn size={14} />}>Zoom in</Button>
        <Button variant="ghost" size="sm" onClick={fitAll} ariaLabel="Fit all stations" leftIcon={<Maximize2 size={14} />}>Fit all</Button>
      </div>
    </div>
    <p className="sg-network-overview__desc"><MapPinned size={14} /> Standard map view — wheel to zoom geographically, drag to pan, and select a station marker.</p>
    <div className="sg-network-overview__map-shell">
      <MapContainer ref={mapRef} center={[22.8, 79.5]} zoom={5} minZoom={3} maxZoom={18} scrollWheelZoom className="sg-network-overview__map" aria-label="Map of SkyGuard weather stations">
        <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <StationViewport stations={validStations} selectedStation={selectedStation} /><ZoomReporter onZoomChange={setZoom} />
        {validStations.map((station) => {
          const selected = selectedStation?.station_id === station.station_id;
          return <CircleMarker key={station.station_id} center={[station.lat, station.lon]} radius={selected ? 10 : 7} pathOptions={{ color: selected ? '#ffffff' : '#07111f', weight: selected ? 3 : 2, fillColor: statusColor(station.status), fillOpacity: 1 }} eventHandlers={{ click: () => onSelectStation(station) }}>
            <LeafletTooltip direction="top" offset={[0, -8]} opacity={1} className="sg-network-overview__marker-label"><strong>{stationCity(station)}</strong><span>{station.station_id} · {station.status}</span></LeafletTooltip>
          </CircleMarker>;
        })}
      </MapContainer>
    </div>
    <div className="sg-network-overview__legend">{[['#10b981', 'Normal'], ['#f59e0b', 'Warning'], ['#ef4444', 'Critical'], ['#64748b', 'Offline']].map(([color, label]) => <span className="sg-network-overview__legend-item" key={label}><i style={{ background: color }} />{label}</span>)}</div>
  </Card>;
};
